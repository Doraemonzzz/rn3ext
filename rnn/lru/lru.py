import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
# reference: https://github.com/bojone/rnn/blob/main/lru.py
from torch import Tensor, nn

from ..helpers import get_activation_fn, print_params


class Lru(nn.Module):
    def __init__(
        self, embed_dim, act_fun="silu", causal=True,
    ):
        self.embed_dim = embed_dim
    
        nu_log, theta_log, gamma_log = self.initializer()
        self.nu_log = nn.Parameter(nu_log, requires_grad=True)
        self.theta_log = nn.Parameter(theta_log, requires_grad=True)
        self.gamma_log = nn.Parameter(gamma_log, requires_grad=True)
        
        self.input_proj = nn.Linear(embed_dim, 2 * embed_dim)
        self.output_proj = nn.Linear(2 * embed_dim, embed_dim)
    
    def initializer(self):
        r_min, r_max = 0.9, 0.999
        u1 = np.random.random(self.embed_dim)
        u2 = np.random.random(self.embed_dim)
        nu_log = np.log(
            -0.5 * np.log(u1 * (r_max**2 - r_min**2) + r_min**2)
        )
        theta_log = np.log(u2 * np.pi * 2)
        gamma_log = np.log(np.sqrt(1 - np.exp(-np.exp(nu_log))**2))
        
        return nu_log, theta_log, gamma_log
    
    def forward(self, x):
        n, b, d = x.shape
        input_state = self.input_proj(x)
        
        nu = torch.exp(self.nu_log)
        theta = torch.exp(self.theta_log) 
        gamma = torch.exp(-torch.exp(self.gamma_log))
        
        gamma_real = nu * torch.cos(theta)
        gamma_imag = nu * torch.sin(theta)
        
        input_state = rearrange(input_state, "n b (e k) -> n b e k", k=2)
        input_real = gamma * input_state[..., 0]
        input_imag = gamma * input_state[..., 1]
        
        hidden_real = torch.zeros(1, b, d).to(x)
        hidden_imag = torch.zeros(1, b, d).to(x)
        hiddens_real = []
        hiddens_imag = []
        for i in range(n):
            # to do, fix index
            hidden_real_next = gamma_real[i] * hidden_real - gamma_imag[i] * hidden_imag + input_real[i]
            hidden_imag_next = gamma_real[i] * hidden_imag + gamma_imag[i] * hidden_real + input_imag[i]
            # print(hidden_real_next.shape)
            hiddens_real.append(hidden_real_next)
            hiddens_imag.append(hidden_imag_next)
            hidden_real = hidden_real_next
            hidden_imag = hidden_imag_next
        hiddens_real = torch.cat(hiddens_real, dim=0)
        hiddens_imag = torch.cat(hiddens_imag, dim=0)
        feature = torch.cat([hiddens_real, hiddens_imag], dim=-1)

        output = self.out_proj(feature)
        
        return output
