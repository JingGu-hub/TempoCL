import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import argparse
import numpy as np

from models.CNN import CNN
from models.MvTCL import MvTCL
from models.TASS import TASS

from utils.data_utils import build_dataset, load_loader
from utils.loss_utils import SemiLoss

from utils.utils import set_seed, f1_scores, adjust_param

import warnings
warnings.filterwarnings("ignore")


# Training
def train(args, epoch, net, net2, temporal_contr, optimizer, labeled_trainloader, unlabeled_trainloader, criterion, warm_up, ):
    net.train()
    net2.eval() #fix one network and train the other

    train_loss = 0
    all_logits_u = torch.zeros((len(unlabeled_trainloader.dataset), 128)).cuda()

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, labels_x, w_x, index) in enumerate(labeled_trainloader):
        unlabeled_train_iter = iter(unlabeled_trainloader)
        inputs_u, u_index = next(unlabeled_train_iter)
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, labels_x, w_x = inputs_x.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u = inputs_u.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11, _, _ = net(inputs_u)
            outputs_u21, _, _ = net2(inputs_u)
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u21, dim=1)) / 2
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            outputs_x, _, _ = net(inputs_x)
            
            px = torch.softmax(outputs_x, dim=1)
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T) # temparature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
        
        # mixmatch
        all_inputs = torch.cat([inputs_x, inputs_u], dim=0)
        all_targets = torch.cat([targets_x, targets_u], dim=0)

        logits, features, _ = net(all_inputs)
        logits_x = logits[:batch_size]
        logits_u = logits[batch_size:]
        all_logits_u[u_index] = features[batch_size:]

        Lx, Lu, lamb = criterion(args, logits_x, all_targets[:batch_size], logits_u, all_targets[batch_size:], epoch+batch_idx/num_iter, warm_up)
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        avg_tcl_loss, multi_scale_loss = temporal_contr(inputs_x, all_targets[:batch_size], net)
        loss = Lx + lamb * Lu + penalty + args.multi_loss_param * avg_tcl_loss + args.tc_param * multi_scale_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss /= num_iter

    return train_loss, all_logits_u

def warmup(net, dataloader, optimizer, CEloss, num_classes=10, mask=None):
    net.train()

    train_loss = 0
    features_all = torch.zeros([len(dataloader.dataset), 128]).cuda()
    pred_all = torch.zeros([len(dataloader.dataset), num_classes]).cuda()
    for i, (inputs, labels, index) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        mask_i = torch.from_numpy(mask[index]).cuda()

        optimizer.zero_grad()
        outputs, features, _ = net(inputs)
        loss = torch.mean(mask_i * CEloss(outputs, labels))

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * labels.size(0)

        features_all[index] = features
        pred_all[index] = outputs
    train_loss /= len(dataloader.dataset)

    return train_loss, features_all, pred_all

def test(test_loader, net1, net2):
    net1.eval()
    net2.eval()

    correct, total, test_loss = 0, 0, 0
    f1_macro, f1_weighted, f1_micro, test_num = 0, 0, 0, 0
    for batch_idx, (inputs, targets, index) in enumerate(test_loader):
        with torch.no_grad():
            test_num += 1

            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1, _, _ = net1(inputs)
            outputs2, _, _ = net2(inputs)

            outputs = (outputs1+outputs2) / 2
            _, predicted = torch.max(outputs, 1)
            loss = F.cross_entropy(outputs, targets)
            test_loss += loss.item()*targets.size(0)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()

            f1_mac, f1_w, f1_mic = f1_scores(outputs, targets)
            f1_macro += f1_mac
            f1_weighted += f1_w
            f1_micro += f1_mic

    test_acc = 100.*correct/total
    test_loss /= total

    f1_macro, f1_weighted, f1_micro = f1_macro/test_num, f1_weighted/test_num, f1_micro/test_num

    return test_acc, test_loss, f1_macro, f1_weighted, f1_micro

def eval_train(args, eval_loader, model, CE, num_classes):
    model.eval()

    losses = torch.zeros(len(eval_loader.dataset)).cuda()
    features_all = torch.zeros([len(eval_loader.dataset), 128]).cuda()
    pred_all = torch.zeros([len(eval_loader.dataset), num_classes]).cuda()
    train_labels_all = torch.zeros(len(eval_loader.dataset)).type(torch.LongTensor).cuda()
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs, features, _ = model(inputs)
            loss = CE(outputs, targets)  

            losses[index]=loss
            features_all[index] = features
            pred_all[index] = outputs
            train_labels_all[index] = targets
    losses = (losses-losses.min())/(losses.max()-losses.min())

    if args.label_noise_rate==0.9: # average loss over last 5 epochs to improve convergence stability
        history = losses
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1,1)
    else:
        input_loss = losses.reshape(-1,1)

    return input_loss.reshape(-1,1).detach().cpu().numpy(), features_all, pred_all, train_labels_all

def main():
    parser = argparse.ArgumentParser()

    # Dataset configuration
    parser.add_argument('--archive', type=str, default='UEA', help='Name of the archive (e.g., UEA, other)')
    parser.add_argument('--dataset', default='ArticularyWordRecognition', type=str, help='Name of the main dataset')
    parser.add_argument('--corruption_dataset', default='InsectSound', type=str, help='Name of the dataset used for corruption/out-of-distribution')
    parser.add_argument('--data_dir', type=str, default='../data/Multivariate2018_arff/Multivariate_arff', help='Directory containing the main dataset')
    parser.add_argument('--corruption_data_dir', type=str, default='../data/ts_noise_data/', help='Directory containing corrupted time series data')

    # Noise label setting
    parser.add_argument('--noise_type', type=str, default='symmetric', help='Type of label noise: symmetric, asymmetric, instance, or pairflip')
    parser.add_argument('--label_noise_rate', type=float, default=0.2, help='Ratio of label noise to apply')
    parser.add_argument('--ood_noise_rate', type=float, default=0.4, help='Ratio of out-of-distribution noise to apply')

    # Basic Training Settings
    parser.add_argument('--warm_up', type=int, default=30, help='Number of warm-up epochs before full training')
    parser.add_argument('--num_epochs', default=200, type=int, help='Total number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training and evaluation')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of subprocesses for data loading')
    parser.add_argument('--seed', default=42, help='Random seed for reproducibility')
    parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='Initial learning rate')
    parser.add_argument('--gpuid', default=0, type=int, help='GPU device ID to use')

    # Semi-supervised setting
    parser.add_argument('--lambda_u', default=25, type=float, help='Weight for unsupervised loss term')
    parser.add_argument('--T', default=0.5, type=float, help='Temperature parameter for sharpening')

    # TempoCL
    parser.add_argument('--multiscale', type=int, default=2, help='Number of scales in multi-scale processing')
    parser.add_argument('--multi_loss_param', type=float, default=0.05, help='Loss weight for multi-scale component')
    parser.add_argument('--tc_param', type=float, default=0.05, help='Weight for temporal contrastive loss')

    args = parser.parse_args()

    torch.cuda.set_device(args.gpuid)
    set_seed(args)
    train_loader, test_loader, train_dataset, train_target, train_noisy_target, test_dataset, test_target, input_channel, seq_len, num_classes, ood_ids = build_dataset(args)
    args.num_class = num_classes

    net1 = CNN(input_channel=input_channel, n_outputs=num_classes).cuda()
    net2 = CNN(input_channel=input_channel, n_outputs=num_classes).cuda()
    temporal_contr1 = MvTCL(seq_len=seq_len, down_sampling_layers=args.multiscale).cuda()
    temporal_contr2 = MvTCL(seq_len=seq_len, down_sampling_layers=args.multiscale).cuda()
    tass = TASS(args)
    cudnn.benchmark = True

    criterion = SemiLoss()
    optimizer1 = optim.SGD([{'params': net1.parameters()}, {'params': temporal_contr1.parameters()}], lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer2 = optim.SGD([{'params': net2.parameters()}, {'params': temporal_contr2.parameters()}], lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler1, scheduler2 = adjust_param(args, optimizer1, optimizer2)

    CE = torch.nn.CrossEntropyLoss(reduction='none')

    ood_mask1 = np.ones(len(train_dataset))
    ood_mask2 = np.ones(len(train_dataset))
    last_five_accs, last_five_losses, last_five_f1_macro, last_five_f1_weighted, last_five_f1_micro = [], [], [], [], []
    for epoch in range(args.num_epochs + 1):
        scheduler1.step()
        scheduler2.step()

        input_loss1, features1, pred1, train_labels_all1 = eval_train(args, train_loader, net1, CE, num_classes)
        input_loss2, features2, pred2, train_labels_all2 = eval_train(args, train_loader, net2, CE, num_classes)

        if epoch < args.warm_up:
            net1_train_loss, _, _ = warmup(net1, train_loader, optimizer1, CE, num_classes, ood_mask1)
            net2_train_loss, _, _ = warmup(net2, train_loader, optimizer2, CE, num_classes, ood_mask2)
        else:
            pred1, prob1, u_ids1 = tass.first_sample_selection(input_loss1)
            pred2, prob2, u_ids2 = tass.first_sample_selection(input_loss2)

            ood_mask1 = tass.second_sample_selection(features1, train_labels_all1, u_ids1)
            ood_mask2 = tass.second_sample_selection(features2, train_labels_all2, u_ids2)

            labeled_trainloader = load_loader(args, train_dataset, train_target, noisy_target=train_noisy_target, pred=pred1, prob=prob1, ood_mask=ood_mask1, ood_ids=ood_ids, mode='labeled')  # co-divide
            unlabeled_trainloader = load_loader(args, train_dataset, train_target, noisy_target=train_noisy_target, pred=pred1, prob=prob1, ood_mask=ood_mask1, ood_ids=ood_ids, mode='unlabeled')  # co-divide
            net1_train_loss, features1 = train(args, epoch, net1, net2, temporal_contr1, optimizer1, labeled_trainloader, unlabeled_trainloader, criterion, args.warm_up)  # train net1

            labeled_trainloader = load_loader(args, train_dataset, train_target, noisy_target=train_noisy_target, pred=pred1, prob=prob1, ood_mask=ood_mask2, ood_ids=ood_ids, mode='labeled')  # co-divide
            unlabeled_trainloader = load_loader(args, train_dataset, train_target, noisy_target=train_noisy_target, pred=pred2, prob=prob2, ood_mask=ood_mask2, ood_ids=ood_ids, mode='unlabeled')  # co-divide
            net2_train_loss, features2 = train(args, epoch, net2, net1, temporal_contr2, optimizer2, labeled_trainloader, unlabeled_trainloader, criterion, args.warm_up)  # train net2

        train_loss = (net1_train_loss + net2_train_loss) / 2
        test_acc, test_loss, f1_macro, f1_weighted, f1_micro = test(test_loader, net1, net2)

        print('Epoch:[%d/%d], train_loss:%.4f, test_loss:%.4f, test_acc:%.4f, f1_macro:%.4f, f1_weighted:%.4f, f1_micro:%.4f' %
              (epoch + 1, args.num_epochs, train_loss, test_loss, test_acc, f1_macro, f1_weighted, f1_micro))
        if (epoch + 5) >= args.num_epochs:
            last_five_accs.append(test_acc)
            last_five_losses.append(test_loss)
            last_five_f1_macro.append(f1_macro)
            last_five_f1_weighted.append(f1_weighted)
            last_five_f1_micro.append(f1_micro)

    test_accuracy, test_loss = round(np.mean(last_five_accs), 4), round(np.mean(last_five_losses), 4)
    f1_macro, f1_weighted, f1_micro = round(np.mean(last_five_f1_macro), 4), round(np.mean(last_five_f1_weighted), 4), round(np.mean(last_five_f1_micro), 4)
    print('Test Accuracy:', test_accuracy, 'Test Loss:', test_loss, 'F1_macro:', f1_macro, 'F1_weighted:', f1_weighted, 'F1_micro:', f1_micro)


if __name__ == '__main__':
    main()

