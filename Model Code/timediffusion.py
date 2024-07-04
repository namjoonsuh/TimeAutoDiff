from typing import List, Callable
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_betas(steps):
    beta_start, beta_end = 1e-4, 0.2
    diffusion_ind = torch.linspace(0, 1, steps).to(device)
    return beta_start * (1 - diffusion_ind) + beta_end * diffusion_ind

diffusion_steps = 100
betas = get_betas(diffusion_steps)
alphas = torch.cumprod(1 - betas, dim=0)

# gp_sigma = 0.015 works pretty well for stock_data
gp_sigma = 0.02

def get_gp_covariance(t):
    s = t - t.transpose(-1, -2)
    diag = torch.eye(t.shape[-2]).to(t) #* 1e-5 # for numerical stability
    return diag #torch.exp(-torch.square(s) / gp_sigma) + diag

def add_noise(x, t, i):
    """
    x: Clean data sample, shape [B, S, D]
    t: Times of observations, shape [B, S, 1]
    i: Diffusion step, shape [B, S, 1]
    """
    noise_gaussian = torch.randn_like(x)
    
    cov = get_gp_covariance(t)
    L = torch.linalg.cholesky(cov)
    
    noise = L @ noise_gaussian
    
    alpha = alphas[i.long()].to(x)
    x_noisy = torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * noise
    
    return x_noisy, noise

#####################################################################################################################

from typing import List, Callable
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_value: float):
        super().__init__()
        self.max_value = max_value

        linear_dim = dim // 2
        periodic_dim = dim - linear_dim

        self.scale = torch.exp(-2 * torch.arange(0, periodic_dim).float() * math.log(self.max_value) / periodic_dim)
        self.shift = torch.zeros(periodic_dim)
        self.shift[::2] = 0.5 * math.pi

        self.linear_proj = nn.Linear(1, linear_dim)

    def forward(self, t):
        periodic = torch.sin(t * self.scale.to(t) + self.shift.to(t))
        linear = self.linear_proj(t / self.max_value)
        return torch.cat([linear, periodic], -1)

class FeedForward(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int, activation: Callable=nn.ReLU(), final_activation: Callable=None):
        super().__init__()

        hidden_dims = hidden_dims[:]
        hidden_dims.append(out_dim)

        layers = [nn.Linear(in_dim, hidden_dims[0])]

        for i in range(len(hidden_dims) - 1):
            layers.append(activation)
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))

        if final_activation is not None:
            layers.append(final_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)    
    
# https://datascience.stackexchange.com/questions/121548/how-to-make-an-rnn-model-in-pytorch-that-has-a-custom-hidden-layers-and-that-i
class BiRNN_score(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, diffusion_steps):
        super(BiRNN_score, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_proj = FeedForward(input_size, [], hidden_size)
        self.t_enc = PositionalEncoding(hidden_size, max_value=1)
        self.i_enc = PositionalEncoding(hidden_size, max_value=diffusion_steps) 
        self.proj = FeedForward(4 * hidden_size, [], hidden_size, final_activation=nn.ReLU())
        
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.layer_norm = nn.LayerNorm(2 * hidden_size)
        self.fc = nn.Linear(2 * hidden_size, input_size)

        self.time_encode = nn.Sequential(nn.Linear(8, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size))
        
    def forward(self, x, t, i, time_info = None):
        shape = x.shape

        x = x.view(-1, *shape[-2:])
        t = t.view(-1, shape[-2], 1)
        i = i.view(-1, shape[-2], 1)
        
        x = self.input_proj(x)
        t = self.t_enc(t)
        i = self.i_enc(i)
        
        if time_info is not None:
            time_info = self.time_encode(time_info)
        
        x = self.proj(torch.cat([x, t, i, time_info], -1))        
        
        out, _ = self.lstm(x)
        output = self.layer_norm(out)
        final_out = self.fc(output)
        
        return final_out

#####################################################################################################################
class EMA:
    def __init__(self, beta):
        self.beta = beta
        self.step = 0
    
    def update_average(self, old, new):
        return self.beta * old + (1-self.beta) * new
    
    def update_model_average(self, ema_model, model):
        for current_param, ema_param in zip(model.parameters(), ema_model.parameters()):
            old_weight, new_weight = ema_param.data, current_param.data
            ema_param.data = self.update_average(old_weight, new_weight)
    
    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())
    
    def step_ema(self, ema_model, model, step_start_ema = 2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

#####################################################################################################################
def get_loss(model, x, t, time_info = None):
    i = torch.randint(0, diffusion_steps, size=(x.shape[0],))
    i = i.view(-1, 1, 1).expand_as(x[...,:1]).to(x)
    
    x_noisy, noise = add_noise(x, t, i)
    pred_noise = model(x_noisy, t, i, time_info)
    
    loss = (pred_noise - noise)**2
    return torch.mean(loss)

#####################################################################################################################
import copy
import tqdm.notebook
import random

def train_diffusion(latent_features, time_info, hidden_dim, num_layers, diffusion_steps, n_epochs):
    input_size = latent_features.shape[2]
    model = BiRNN_score(input_size, hidden_dim, num_layers, diffusion_steps).to(device)
    optim = torch.optim.Adam(model.parameters())
    ema = EMA(beta=0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    x = latent_features.detach().to(device); 
    N, T, K = latent_features.shape

    batch_size = diffusion_steps

    all_indices = list(range(x.shape[0]))
    tqdm_epoch = tqdm.notebook.trange(n_epochs)

    for epoch in tqdm_epoch:
        batch_indices = random.sample(all_indices, batch_size)
        optim.zero_grad()
        t = torch.rand(diffusion_steps, T, 1).sort(1)[0].to(device)
        loss = get_loss(model, x[batch_indices,:,:], t, time_info[batch_indices,:,:])
        loss.backward()
        optim.step()
        ema.step_ema(ema_model, model)

        tqdm_epoch.set_description('Average Loss: {:5f}'.format(loss.item()))
    
    return model

#####################################################################################################################
@torch.no_grad()
def sample(t,emb,model,time_info):
    cov = get_gp_covariance(t)
    L = torch.linalg.cholesky(cov)
    
    x = L @ torch.randn_like(emb)
    
    for diff_step in reversed(range(0, diffusion_steps)):
        alpha = alphas[diff_step]
        beta = betas[diff_step]

        z = L @ torch.randn_like(emb)

        i = torch.Tensor([diff_step]).expand_as(x[...,:1]).to(device)
        pred_noise = model(x, t, i, time_info)
        
        x = (1/(1 - beta).sqrt()) * (x - beta * pred_noise / (1 - alpha).sqrt()) + beta.sqrt() * z
    return x






