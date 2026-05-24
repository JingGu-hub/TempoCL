import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x

class Seq_Transformer(nn.Module):
    def __init__(self, *, patch_size, dim, depth, heads, mlp_dim, channels=1, dropout=0.1):
        super().__init__()
        patch_dim = channels * patch_size
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.c_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_c_token = nn.Identity()

    def forward(self, forward_seq):
        x = self.patch_to_embedding(forward_seq)
        b, n, _ = x.shape
        c_tokens = repeat(self.c_token, '() n d -> b n d', b=b)
        x = torch.cat((c_tokens, x), dim=1)
        x = self.transformer(x)
        c_t = self.to_c_token(x[:, 0])
        return c_t

class temporal_contrast(nn.Module):
    def __init__(self, final_out_channels=128, hidden_dim=128, timesteps=8):
        super(temporal_contrast, self).__init__()
        self.num_channels = final_out_channels
        self.lsoftmax = nn.LogSoftmax()

        self.long_timestep = timesteps
        self.short_timestep = max(int(timesteps / 2), 1)
        self.long_pred = nn.ModuleList([nn.Linear(hidden_dim, self.num_channels) for i in range(self.long_timestep)]).cuda()
        self.short_pred = nn.ModuleList([nn.Linear(hidden_dim, self.num_channels) for i in range(self.short_timestep)]).cuda()

        self.seq_transformer = Seq_Transformer(patch_size=self.num_channels, dim=hidden_dim, depth=4, heads=4, mlp_dim=64).cuda()

        self.long_predict_downsampling = torch.nn.Sequential(
            torch.nn.Linear(self.long_timestep, self.long_timestep),
            torch.nn.GELU(),
            torch.nn.Linear(self.long_timestep, self.short_timestep)
        )
        self.short_predict_upsampling = torch.nn.Sequential(
            torch.nn.Linear(self.short_timestep, self.short_timestep),
            torch.nn.GELU(),
            torch.nn.Linear(self.short_timestep, self.long_timestep)
        )
        self.mid_predict_downsampling = torch.nn.Sequential(
            torch.nn.Linear(self.long_timestep, self.long_timestep),
            torch.nn.GELU(),
            torch.nn.Linear(self.long_timestep, self.long_timestep)
        )

    def similarity(self, z, z_prime):
        z = z.view(z.shape[0], -1)
        z_prime = z_prime.view(z_prime.shape[0], -1)
        return torch.sum(z * z_prime, dim=-1) / (torch.norm(z, dim=-1) * torch.norm(z_prime, dim=-1))

    def infoNCE(self, pos_pair, neg_pair, temperature=0.05):
        pos_timestep, batch, _ = pos_pair.shape
        neg_timestep, _, _ = neg_pair.shape
        pos_timestep, neg_timestep = pos_timestep //2, neg_timestep // 2
        timestep = min(pos_timestep, neg_timestep)

        z_pos, z_pos_list = pos_pair[:pos_timestep, :, :], pos_pair[pos_timestep:, :, :]
        z_neg, z_neg_list = neg_pair[:neg_timestep, :, :], neg_pair[neg_timestep:, :, :]

        infonce = 0
        for i in range(timestep):
            if pos_timestep > neg_timestep:
                exp_pos = torch.sum(torch.stack([
                    torch.exp(self.similarity(z_pos[2 * i + j], z_pos_list[2 * i + j]) / temperature) for j in range(2)
                ]), dim=0)
            else:
                sim_pos = self.similarity(z_pos[i], z_pos_list[i]) / temperature
                exp_pos = torch.exp(sim_pos)

            if neg_timestep > pos_timestep:
                exp_neg = torch.sum(torch.stack([
                    torch.exp(self.similarity(z_neg[2 * i + j], z_neg_list[2 * i + j]) / temperature) for j in range(2)
                ]), dim=0)
            else:
                sim_neg = self.similarity(z_neg[i], z_neg_list[i]) / temperature
                exp_neg = torch.exp(sim_neg)

            numerator = exp_pos
            denominator = exp_pos + exp_neg
            infonce += -torch.sum(torch.log(numerator / denominator))
        infonce /= batch * timestep

        return infonce

    def forward(self, z_aug1, z_aug2, pred_type='long'):
        z_aug1 = z_aug1.transpose(1, 2)
        z_aug2 = z_aug2.transpose(1, 2)

        batch = z_aug1.shape[0]
        if pred_type == 'long':
            pred_timestep = self.long_timestep
            t_samples = torch.randint(low=pred_timestep-1, high=z_aug1.shape[-2] // 2, size=(1,)).long().cuda()  # randomly pick time stamps
            pred_linear = self.long_pred
            pred_sampling = self.long_predict_downsampling

            start_time, timestep = t_samples // 2, self.short_timestep
            encode_samples = torch.empty((self.short_timestep, batch, self.num_channels)).float().cuda()
        elif pred_type == 'short':
            pred_timestep = self.short_timestep
            t_samples = torch.randint(low=pred_timestep-1, high=z_aug1.shape[-2] // 2, size=(1,)).long().cuda()
            pred_linear = self.short_pred
            pred_sampling = self.short_predict_upsampling

            start_time, timestep = t_samples * 2, self.long_timestep
            encode_samples = torch.empty((self.long_timestep, batch, self.num_channels)).float().cuda()
        elif pred_type == 'mid':
            pred_timestep = self.long_timestep
            t_samples = torch.randint(low=pred_timestep - 1, high=z_aug1.shape[-2] // 2, size=(1,)).long().cuda()  # randomly pick time stamps
            pred_linear = self.long_pred
            # pred_sampling = self.mid_predict_downsampling

            start_time, timestep = t_samples, self.long_timestep
            encode_samples = torch.empty((self.long_timestep, batch, self.num_channels)).float().cuda()
        else:
            raise NotImplementedError

        for i in np.arange(1, timestep + 1):
            encode_samples[i - 1] = z_aug2[:, start_time + i, :].view(batch, self.num_channels)
        forward_seq = z_aug1[:, t_samples+1-pred_timestep:t_samples+1, :]
        target_seq = z_aug1[:, t_samples+1:t_samples+1+pred_timestep, :]

        c_t = self.seq_transformer(forward_seq)
        target_pred = torch.empty((pred_timestep, batch, self.num_channels)).float().cuda()
        for i in np.arange(0, pred_timestep):
            target_pred[i] = pred_linear[i](c_t)

        target_seq = target_seq.transpose(0, 1)
        pos_pair = torch.cat((target_seq, target_pred), dim=0)

        sampling_pred = rearrange(target_pred, 't b c -> b c t')
        sampling_pred = pred_sampling(sampling_pred) if pred_type != 'mid' else sampling_pred
        sampling_pred = rearrange(sampling_pred, 'b c t-> t b c')
        neg_pair = torch.cat((encode_samples, sampling_pred), dim=0)

        info_nce = self.infoNCE(pos_pair, neg_pair)
        pos_mse_loss = F.mse_loss(target_pred, target_seq)
        loss = info_nce + pos_mse_loss
        return loss

class MvTCL(nn.Module):

    def __init__(self, seq_len, down_sampling_layers=2, down_sampling_method='avg', down_sampling_window=2, pred_len=8):
        super(MvTCL, self).__init__()

        self.down_sampling_method = down_sampling_method
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_window = down_sampling_window

        final_length = max(int(seq_len / (2 ** 2 * 2)), 1)
        timesteps = pred_len if final_length > pred_len else final_length
        self.time_constrast_blocks = nn.ModuleList([temporal_contrast(timesteps=int(timesteps / (2 ** i))) for i in range(max(down_sampling_layers-1, 1))])

    def multi_scale_down_sampling(self, x_enc):
        if self.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.down_sampling_window, return_indices=False)
        elif self.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.down_sampling_window)
        else:
            return x_enc

        x_enc_ori = x_enc
        x_enc_sampling_list = []
        x_enc_sampling_list.append(x_enc)
        for i in range(self.down_sampling_layers - 1):
            x_enc_sampling = down_pool(x_enc_ori)

            if x_enc_sampling.shape[2] > 1:
                x_enc_sampling_list.append(x_enc_sampling)
            x_enc_ori = x_enc_sampling

        return x_enc_sampling_list

    def forward(self, x, target, model):
        x_enc_list = self.multi_scale_down_sampling(x)

        logits_list, scale_features_list = [], []
        for i, x_enc in zip(range(len(x_enc_list)), x_enc_list):
            logits, _, features = model(x_enc)
            features = F.normalize(features, dim=1)
            scale_features_list.append(features)
            logits_list.append(logits)

        multi_scale_logits = 0
        for i in range(len(logits_list)):
            multi_scale_logits += logits_list[i]
        multi_scale_loss = -torch.mean(torch.sum(F.log_softmax(multi_scale_logits, dim=1) * target, dim=1))

        neg_tcl_list = []
        if self.down_sampling_layers == 1:
            orin_neg_loss = self.time_constrast_blocks[0](scale_features_list[0], scale_features_list[0], pred_type='mid')
            neg_tcl_list.append(orin_neg_loss)

        for i in range(len(scale_features_list) - 1):
            if scale_features_list[i+1].shape[2] > 1:
                neg_loss1 = self.time_constrast_blocks[i](scale_features_list[i], scale_features_list[i+1], pred_type='long')
                neg_tcl_list.append(neg_loss1)
            if scale_features_list[i].shape[2] > 1:
                neg_loss2 = self.time_constrast_blocks[i](scale_features_list[i+1], scale_features_list[i], pred_type='short')
                neg_tcl_list.append(neg_loss2)

        avg_ntcl_loss = sum(neg_tcl_list) / len(neg_tcl_list)

        return avg_ntcl_loss, multi_scale_loss