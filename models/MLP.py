import math
import copy
from pathlib import Path
from functools import partial
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

class MLP(nn.Module):
    """
    MLP for the reverse diffusion process
    """
    def __init__ (
                  self, 
                  in_dims, 
                  time_emb_dim, 
                  tag_emb_dim, 
                  channels=None, 
                  dim_type='cat',
                  act_func='tanh',
                  learned_sinusoidal_cond = False,
                  learned_sinusoidal_dim=8,
                  random_fourier_features=False, 
                  norm=False, 
                  dropout=0.5,
        ):
        
        super().__init__()

        self.model_type = MLP
        self.in_dims = in_dims
        self.out_dims = self.in_dims[::-1]
        assert self.out_dims[0] == self.in_dims[-1], "In and out dimensions must equal to each other."
        self.norm = norm
        self.time_emb_dim = time_emb_dim
        self.tag_emb_dim = tag_emb_dim
        self.channels = channels
        self.dim_type = dim_type
        self.learned_sinusoidal_dim = learned_sinusoidal_dim
        self.dropout = dropout
        self.num_layers = len(self.in_dims) - 1

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(in_dims[0])
            fourier_dim = in_dims[0]

        self.time_embedding = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.tag_embedding = nn.Sequential(nn.Linear(tag_emb_dim, in_dims[0]))

        if self.dim_type == "cat":
            # ex) [item_emb(128) + time_emb(10) + tag_emb(128), item_emb(128)]
            self.in_dims[0] = self.in_dims[0]*2 + self.time_emb_dim
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.dim_type)
        
        layers = []
        for i in range(self.num_layers - 1):
            layers.append(nn.Linear(self.in_dims[i], self.in_dims[i+1]))
            if act_func == 'tanh':
                layers.append(nn.Tanh())
            elif act_func == 'relu':
                layers.append(nn.ReLU())
            elif act_func == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif act_func == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            else:
                raise ValueError("Unsupported activation function %s" % act_func)
            layers.append(nn.Dropout(self.dropout))
         
        layers.append(nn.Linear(self.in_dims[self.num_layers-1], self.out_dims[0]))

        self.input_dropout = nn.Dropout(self.dropout)
        self.mlp = nn.Sequential(*layers)

        self.init_weights()

    def init_weights(self):
        def initialize_layer(layer):
            if isinstance(layer, nn.Linear):
                fan_in, fan_out = layer.weight.size(1), layer.weight.size(0)
                std = np.sqrt(2.0 / (fan_in + fan_out))
                layer.weight.data.normal_(0.0, std)
                layer.bias.data.normal_(0.0, 0.001)

        # Initialize the time_embedding layers
        for layer in self.time_embedding:
            initialize_layer(layer)
        
        # Initialize the tag_embedding layers
        for layer in self.tag_embedding:
            initialize_layer(layer)

        # Initialize the MLP layers
        for layer in self.mlp:
            initialize_layer(layer)

    def forward(self, x, timesteps, tag):
        time_emb = self.time_embedding(timesteps).to(x.device)
        tag_emb = self.tag_embedding(tag).to(x.device)
        if self.norm:
            x = F.normalize(x)

        x = self.input_dropout(x)
        # tag_emb = self.input_dropout(tag_emb)

        h = torch.cat([x, time_emb, tag_emb], dim=-1)
        return self.mlp(h)