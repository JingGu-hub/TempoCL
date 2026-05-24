import datetime
import os
import random

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture

from cleanlab.internal.constants import EPSILON

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def create_file(path, filename, write_line=None, exist_create_flag=True):
    create_dir(path)
    filename = os.path.join(path, filename)

    if filename != None:
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        if not os.path.exists(filename):
            with open(filename, "a") as myfile:
                print("create new file: %s" % filename)
            with open(filename, "a") as myfile:
                myfile.write(write_line + '\n')
        elif exist_create_flag:
            new_file_name = filename + ".bak-%s" % nowTime
            os.system('mv %s %s' % (filename, new_file_name))
            with open(filename, "a") as myfile:
                myfile.write(write_line + '\n')

    return filename

def f1_scores(output, y_true):
    target_pred = torch.argmax(output.data, axis=1)

    y_true = y_true.detach().cpu().numpy()
    y_pred = target_pred.detach().cpu().numpy()

    # 计算F1分数
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    return f1_macro, f1_weighted, f1_micro

class CustomMultiStepLR:
    def __init__(self, optimizer, milestones, gammas):
        self.optimizer = optimizer
        self.milestones = milestones
        self.gammas = gammas
        self.current_epoch = 0
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self):
        self.current_epoch += 1
        for i, milestone in enumerate(self.milestones):
            if self.current_epoch == milestone:
                self.optimizer.param_groups[0]['lr'] *= self.gammas[i]

def compute_similarity(features, temperature=0.05):
    # compute logits
    anchor_dot_contrast = torch.div(torch.matmul(features, features.T), temperature)
    anchor_dot_contrast[anchor_dot_contrast == float('inf')] = 1

    # for numerical stability
    pos_similarity_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    similarity = anchor_dot_contrast - pos_similarity_max.detach()
    similarity[similarity == float('inf')] = 1

    return similarity

def adjust_param(args, optimizer1, optimizer2):
    args.use_gmm_divide_strategy = True
    scheduler1 = CustomMultiStepLR(optimizer1, milestones=[30, 50, 100, 150], gammas=[0.05, 0.1, 0.1, 0.1])
    scheduler2 = CustomMultiStepLR(optimizer2, milestones=[30, 50, 100, 150], gammas=[0.05, 0.1, 0.1, 0.1])

    if args.dataset in ['ArticularyWordRecognition', 'HAR']:
        scheduler1 = CustomMultiStepLR(optimizer1, milestones=[100, 150], gammas=[0.5, 0.1])
        scheduler2 = CustomMultiStepLR(optimizer2, milestones=[100, 150], gammas=[0.5, 0.1])
    if args.dataset in ['HAR', 'FaceDetection']:
        args.use_gmm_divide_strategy = False

    # args.multiscale = 1 if args.dataset == 'PenDigits' else 2

    return scheduler1, scheduler2

