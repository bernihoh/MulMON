import math
import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F
import pdb
from models import KMeans

def to_sigma(logvar):
    """ Compute std """
    return torch.exp(0.5*logvar)


def layernorm(x):
    """
    :param x: (B, K, L) or (B, K, C, H, W)
    (function adapted from: https://github.com/MichaelKevinKelly/IODINE)
    """
    if len(x.size()) == 3:
        layer_mean = x.mean(dim=2, keepdim=True)
        layer_std = x.std(dim=2, keepdim=True)
    elif len(x.size()) == 5:
        mean = lambda x: x.mean(2, keepdim=True).mean(3, keepdim=True).mean(4, keepdim=True)
        layer_mean = mean(x)
        # this is not implemented in some version of torch
        layer_std = torch.pow(x - layer_mean, 2)
        layer_std = torch.sqrt(mean(layer_std))
    else:
        assert False, 'invalid size for layernorm'

    x = (x - layer_mean) / (layer_std + 1e-5)
    return x


def kl_exponential(post_mu, post_sigma, z_samples=None, pri_mu=None, pri_sigma=None):
    """Support Gaussian only now"""
    if pri_mu is None:
        pri_mu = torch.zeros_like(post_mu, device=post_mu.device, requires_grad=True)
    if pri_sigma is None:
        pri_sigma = torch.ones_like(post_sigma, device=post_sigma.device, requires_grad=True)
    p_post = dist.Normal(post_mu, post_sigma)
    if z_samples is None:
        z_samples = p_post.rsample()
    p_pri = dist.Normal(pri_mu, pri_sigma)
    return p_post.log_prob(z_samples) - p_pri.log_prob(z_samples)


def Gaussian_ll(x_col, _x, masks, std):
    """
    x_col: [B,K,C,H,W]
    _x:    [B,K,3,H,W]
    masks:   [B,K,1,H,W]
    """
    B, K, _, _, _ = x_col.size()
    std_t = torch.tensor([std] * K, device=x_col.device, dtype=x_col.dtype, requires_grad=False)
    std_t = std_t.expand(1, K).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    log_pxz = dist.Normal(x_col, std_t).log_prob(_x)
    ll_pix = torch.logsumexp((masks + 1e-6).log() + log_pxz, dim=1, keepdim=True)  # [B,K,3,H,W]
    assert ll_pix.min().item() > -math.inf
    return ll_pix, log_pxz


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, image_size):
        super(Encoder, self).__init__()
        height = image_size[0]
        width = image_size[1]
        self.convs = nn.Sequential(
            nn.Conv2d(input_dim, 32, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ReLU(inplace=True)
        )

        for i in range(4):
            width = (width - 1) // 2
            height = (height - 1) // 2

        self.mlp = nn.Sequential(
            nn.Linear(64 * width * height, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.shape[0], -1)
        return self.mlp(x)


class SpatialBroadcastDec(nn.Module):
    """Variational Autoencoder with spatial broadcast decoder"""
    def __init__(self, input_dim, output_dim, image_size, decoder='sbd'):
        super(SpatialBroadcastDec, self).__init__()
        self.height = image_size[0]
        self.width = image_size[1]
        self.convs = nn.Sequential(
            nn.Conv2d(input_dim+2, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, output_dim, 1),
        )
        ys = torch.linspace(-1, 1, self.height + 8)
        xs = torch.linspace(-1, 1, self.width + 8)
        ys, xs = torch.meshgrid(ys, xs)
        coord_map = torch.stack((ys, xs)).unsqueeze(0)
        self.register_buffer('coord_map_const', coord_map)

    def forward(self, z):
        z_tiled = z.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.height + 8, self.width + 8)
        coord_map = self.coord_map_const.repeat(z.shape[0], 1, 1, 1)
        inp = torch.cat((z_tiled, coord_map), 1)
        result = self.convs(inp)
        return result


class RefineNetLSTM(nn.Module):
    """
    function Phi (see Sec 3.3 of the paper)
    (adapted from: https://github.com/MichaelKevinKelly/IODINE)
    """
    def __init__(self, z_dim, channels_in, image_size, kmeans_config):
        super(RefineNetLSTM, self).__init__()
        self.convnet = Encoder(channels_in, 128, image_size)
        self.fc_out = nn.Linear(128, 2 * z_dim)
        self.kmeans_config = kmeans_config
        self.n_clusters = 100
        if kmeans_config in ['init_before_lstm', 'init_after_lstm', 'before_lstm_only', 'after_lstm_only']:
            self.cc_encoding = nn.Sequential(
                nn.Linear(self.n_clusters * 3, self.n_clusters * 3 // 2),
                nn.ReLU(inplace=True),
                nn.Linear(self.n_clusters * 3 // 2, 4 * z_dim))
            if kmeans_config in ['init_before_lstm', 'before_lstm_only']:   # init_only, init_before_lstm, init_after_lstm, before_lstm_only, after_lstm_only
                self.lstm = nn.LSTMCell(128 + 4 * z_dim + 4 * z_dim, 128, bias=True)

            #    self.cc_conv_integrate = nn.Sequential(nn.Linear(2*128, 3*128//2),
            #                                           nn.ReLU(inplace=True),
            #                                           nn.Linear(3*128//2, 128))
            if kmeans_config in ['init_after_lstm', 'after_lstm_only']:
                self.lstm = nn.LSTMCell(128 + 4 * z_dim, 128, bias=True)
                self.cc_h_integrate = nn.Sequential(nn.Linear(128 + 4 * z_dim, (2*128 + 4 * z_dim)//2),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear((2*128 + 4 * z_dim) // 2, 128),
                                                    nn.ReLU(inplace=True))
        else:
            self.lstm = nn.LSTMCell(128 + 4 * z_dim, 128, bias=True)

    def forward(self, x, h, c, cluster_centers):
        x_img, lmbda_moment = x['img'], x['state']
        conv_codes = self.convnet(x_img)
        if self.kmeans_config in ['init_before_lstm', 'init_after_lstm', 'before_lstm_only', 'after_lstm_only']:
            #print("a")
            cc_enc = self.cc_encoding(cluster_centers)
            cc_enc = cc_enc[:, None, ...].repeat(1, conv_codes.shape[0]//cluster_centers.shape[0], 1).flatten(start_dim=0, end_dim=1)
            if self.kmeans_config in ['init_before_lstm', 'before_lstm_only']:
                #print("b")
                conv_codes = torch.cat((conv_codes, cc_enc), dim=1)
        lstm_input = torch.cat((lmbda_moment, conv_codes), dim=1)
        h, c = self.lstm(lstm_input, (h, c))
        if self.kmeans_config in ['init_after_lstm', 'after_lstm_only']:
            #print("c")
            _h = self.cc_h_integrate(torch.cat((h, cc_enc), dim=1))
            return self.fc_out(_h), h, c
        return self.fc_out(h), h, c
