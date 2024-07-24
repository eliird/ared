import torch
import torch.nn as nn
from .embracenet import EmbraceNet
from .transformer import TransformerEncoder

class EmotionNet(nn.Module):
    def __init__(self, hyper_params):
        super(EmotionNet, self).__init__()

        self.out_dim = hyper_params.out_dim
        self.feat_dim = hyper_params.embracenet_feat_dim
        self.embrace = EmbracedNetwork(hyper_params)
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, self.out_dim)
        )

    def forward(self, x_a, x_b, x_c):
        # print("Emotion Net: ", x_a.shape, x_b.shape, x_c.shape)
        embrace = self.embrace(x_a, x_b, x_c)
        out = self.classifier(embrace)
        return out

class EmbracedNetwork(nn.Module):
    def __init__(self, hyper_params):
        super(EmbracedNetwork, self).__init__()
        self.device = hyper_params.device
        self.d_mod_l = int(hyper_params.orig_d_mod_l)
        self.d_mod_v = int(hyper_params.orig_d_mod_v)
        self.d_mod_a = int(hyper_params.orig_d_mod_a)
        self.feat_dim = int(hyper_params.embracenet_feat_dim)
        self.conv_dim = int(hyper_params.conv_dim)
        self.mod1 = CrossModal(self.d_mod_l, self.d_mod_a, self.d_mod_v, hyper_params)
        self.mod2 = CrossModal(self.d_mod_a, self.d_mod_l, self.d_mod_v, hyper_params)
        self.mod3 = CrossModal(self.d_mod_v, self.d_mod_l, self.d_mod_a, hyper_params)

        self.embrace = EmbraceNet(device=self.device, input_size_list=[2*self.conv_dim, 2*self.conv_dim, 2*self.conv_dim], embracement_size=self.feat_dim, )

    def forward(self, x_a, x_b, x_c):
        # print("Embrace Net: ", x_a.shape, x_b.shape, x_c.shape)
        mod1 = self.mod1(x_a, x_b, x_c)
        mod2 = self.mod2(x_b, x_a, x_c)
        mod3 = self.mod3(x_c, x_a, x_b)
        embrace = self.embrace(input_list=[mod1, mod2, mod3])
        return embrace
    
class CrossModal(nn.Module):
    def __init__(self, d_mod_a, d_mod_b, d_mod_c, hyper_params):
        super(CrossModal, self).__init__()
        self.cross_att1 = CrossAttention(d_mod_a, d_mod_b, hyper_params)
        self.cross_att2 = CrossAttention(d_mod_a, d_mod_c, hyper_params)
        self.trans_mem = MixCrossAttention(d_mod_a, hyper_params) 

    def forward(self, x_a, x_b, x_c):
        # print("Cross Modal: ", x_a.shape, x_b.shape, x_c.shape)

        h1 = self.cross_att1(x_a, x_b)
        h2 = self.cross_att2(x_a, x_c)
        h = self.trans_mem(h1, h2)
        return h
    
class CrossAttention(nn.Module):
    def __init__(self, d_mod_a, d_mod_b, hyper_params) -> None:
        super(CrossAttention, self).__init__()
        self.orig_d_mod_a, self.orig_d_mod_b = d_mod_a, d_mod_b
        # print("Cross Attention Instance: ", d_mod_a, d_mod_b)
        self.d_mod_a, self.d_mod_b = int(hyper_params.conv_dim), int(hyper_params.conv_dim)
        self.num_heads = int(hyper_params.num_heads)
        self.layers = int(hyper_params.layers)
        self.attn_dropout = float(hyper_params.attn_dropout)
        self.relu_dropout = float(hyper_params.relu_dropout)
        self.res_dropout = float(hyper_params.res_dropout)
        self.embed_dropout = float(hyper_params.embed_dropout)
        self.attn_mask = bool(hyper_params.attn_mask)
        self.device = hyper_params.device
        self.proj_mod_a = nn.Conv1d(self.orig_d_mod_a, self.d_mod_a, kernel_size=1, padding=0, bias=False)
        self.proj_mod_b = nn.Conv1d(self.orig_d_mod_b, self.d_mod_b, kernel_size=1, padding=0, bias=False)

        self.trans_mod_a_with_mod_b = self.get_network()

    def get_network(self, layers=-1):
        embed_dim, attn_dropout = self.d_mod_a, self.attn_dropout
      
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask,
                                  device=self.device)
    
    def forward(self, x_alpha, x_beta):
        # print("Cross Attention: ", x_alpha.shape, x_beta.shape)
        x_a = x_alpha.transpose(1, 2)
        x_b = x_beta.transpose(1, 2)
        proj_x_alpha = x_a if self.orig_d_mod_a == self.d_mod_a else self.proj_mod_a(x_a)
        proj_x_beta = x_b if self.orig_d_mod_b == self.d_mod_b else self.proj_mod_b(x_b)

        proj_x_alpha = proj_x_alpha.permute(2, 0, 1)
        proj_x_beta = proj_x_beta.permute(2, 0, 1)
        h_ls = self.trans_mod_a_with_mod_b(proj_x_alpha, proj_x_beta, proj_x_beta)    # Dimension (L, N, d_l)

        return h_ls

class MixCrossAttention(nn.Module):
    def __init__(self, d_mod, hyper_params) -> None:
        super(MixCrossAttention, self).__init__()
        self.orig_d_mod = d_mod
        self.d_mod_a, self.d_mod_b = int(hyper_params.conv_dim), int(hyper_params.conv_dim)
        self.num_heads = int(hyper_params.num_heads)
        self.layers = int(hyper_params.layers)
        self.attn_dropout = float(hyper_params.attn_dropout)
        self.relu_dropout = float(hyper_params.relu_dropout)
        self.res_dropout = float(hyper_params.res_dropout)
        self.embed_dropout = float(hyper_params.embed_dropout)
        self.attn_mask = bool(hyper_params.attn_mask)
        self.device = hyper_params.device
        self.trans_mem = self.get_network(layers=3)

    def get_network(self, layers):
        embed_dim, attn_dropout = 2*self.d_mod_a, self.attn_dropout
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask,
                                  device=self.device)


    def forward(self, x_a, x_b):
        h = torch.cat([x_a, x_b], dim=2)
        h = self.trans_mem(h, h, h)
        if type(h) == tuple:
            h = h[0]
        h = h[-1]
        return h
