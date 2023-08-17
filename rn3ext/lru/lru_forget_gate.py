# reference: https://github.com/bojone/rnn/blob/main/lru.py
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from torch import Tensor, nn

from ..helpers import get_activation_fn, print_params
from .cuda import LruFunction

class LruForgetGate(nn.Module):
    def __init__(
        self, embed_dim,
    ):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)
        self.embed_dim = embed_dim
    
        nu_log, theta_log, gamma_log = self.initializer()
        self.theta_log = nn.Parameter(theta_log, requires_grad=True)
        self.gamma_log = nn.Parameter(gamma_log, requires_grad=True)
        
        self.nu_proj = nn.Linear(embed_dim, embed_dim)
        self.in_proj = nn.Linear(embed_dim, 2 * embed_dim)
        self.out_proj = nn.Linear(2 * embed_dim, embed_dim)
    
    def initializer(self):
        r_min, r_max = 0.9, 0.999
        u1 = np.random.random(self.embed_dim)
        u2 = np.random.random(self.embed_dim)
        nu_log = np.log(
            -0.5 * np.log(u1 * (r_max**2 - r_min**2) + r_min**2)
        )
        theta_log = np.log(u2 * np.pi * 2)
        gamma_log = np.log(np.sqrt(1 - np.exp(-np.exp(nu_log))**2))
        
        return torch.Tensor(nu_log), torch.Tensor(theta_log), torch.Tensor(gamma_log)

    def forward(self, x, **kwargs):
        x = x.transpose(1, 0)
        n, b, d = x.shape
        input_state = self.in_proj(x)
        
        # shape, (d, )
        nu = F.sigmoid(self.nu_proj(x))
        theta = torch.exp(self.theta_log) 
        gamma = torch.exp(self.gamma_log)

        lambda_real = nu * torch.cos(theta)
        lambda_imag = nu * torch.sin(theta)
        
        input_state = rearrange(input_state, "n b (e k) -> n b e k", k=2)
        input_real = gamma * input_state[..., 0]
        input_imag = gamma * input_state[..., 1]
        
        hiddens_real, hiddens_imag = LruFunction.apply(input_real, input_imag, lambda_real, lambda_imag)
        feature = torch.cat([hiddens_real, hiddens_imag], dim=-1)

        output = self.out_proj(feature)
        
        output = output.transpose(1, 0)
        
        return output
