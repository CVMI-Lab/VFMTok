import kornia, cv2
import numpy as np
import torch.nn as nn
from typing import Union 
from einops import rearrange
import math, kornia, torch, pdb
from torch.nn import functional as F
from vfmtok.engine.util import disabled_train
import vfmtok.modules.pretrained_enc.models_pretrained_enc as models_pretrained_enc

class SelfSupervisedCondtionEmbedder(nn.Module):

    def __init__(self, pretrained_enc_arch, pretrained_enc_path,
                 pretrained_enc_withproj = False, proj_dim = 768, 
                 pretrained_enc_pca_path = None, antialias = True):

        super().__init__()

        self.antialias = antialias
        if 'dinov2' in pretrained_enc_arch:
            self.pretrained_encoder = models_pretrained_enc.__dict__[pretrained_enc_arch](pretrained=False)
        elif 'moco' in pretrained_enc_arch:
            self.pretrained_encoder = models_pretrained_enc.__dict__[pretrained_enc_arch](pretrained=False,
                proj_dim = proj_dim)
        else:
            raise NotImplementedError
        
        if 'dinov2' in pretrained_enc_arch:
            self.pretrained_encoder = models_pretrained_enc.load_pretrained_dino_v2(self.pretrained_encoder, pretrained_enc_path)
        elif 'moco' in pretrained_enc_arch:
            self.pretrained_encoder = models_pretrained_enc.load_pretrained_moco(self.pretrained_encoder, pretrained_enc_path)
        else:
            raise NotImplementedError

        if pretrained_enc_pca_path is not None:
            pca = np.load(pretrained_enc_pca_path, allow_pickle=True).item()
            self.pca_component = torch.Tensor(pca["components"]).cuda()
            self.pca_mean = torch.Tensor(pca["mean"]).cuda()
            self.pretrained_enc_use_pca = True
        else:
            self.pretrained_enc_use_pca = False

        self.pretrained_encoder.cuda()
        self.pretrained_encoder.eval()
        self.pretrained_encoder.train = disabled_train
        try:
            self.pretrained_enc_withproj = pretrained_enc_withproj
        except:
            self.pretrained_enc_withproj = False
    
    def preprocess(self, x):

        # normalize to [0,1]
        x = kornia.geometry.resize(
            x,
            (336, 336),
            interpolation="bicubic",
            align_corners=True,
            antialias=self.antialias,
        )
        x = x.clamp(min=-1, max=1)
        x = (x + 1.0) / 2.0
        return x

    def forward(self, x):

        x = self.preprocess(x)
        mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        std = torch.Tensor([0.229, 0.224, 0.225]).cuda().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        x_normalized = (x - mean) / std
        output = self.pretrained_encoder.forward_features_levels(x_normalized, levels=[5, 11, 17, 23])

        rep = output['feature_list']
        if self.pretrained_enc_withproj:
            rep = self.pretrained_encoder.head(rep)
        
        return rep