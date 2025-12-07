from typing import List, Callable
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import process_edited as pce
from rich.progress import Progress
import dataprovider as dp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

################################################################################################################
def compute_sine_cosine(v, num_terms):
    num_terms = torch.tensor(num_terms).to(device)
    v = v.to(device)

    # Compute the angles for all terms
    angles = 2**torch.arange(num_terms).float().to(device) * torch.tensor(math.pi).to(device) * v.unsqueeze(-1)

    # Compute sine and cosine values for all angles
    sine_values = torch.sin(angles)
    cosine_values = torch.cos(angles)

    # Reshape sine and cosine values for concatenation
    sine_values = sine_values.reshape(*sine_values.shape[:-2], -1)
    cosine_values = cosine_values.reshape(*cosine_values.shape[:-2], -1)

    # Concatenate sine and cosine values along the last dimension
    result = torch.cat((sine_values, cosine_values), dim=-1)

    return result

################################################################################################################
class Embedding_data(nn.Module):
    def __init__(self, input_size, emb_dim, n_bins, n_cats, n_nums, cards):
        super().__init__()
        
        self.n_bins = n_bins
        self.n_cats = n_cats
        self.n_nums = n_nums
        self.cards = cards
        
        self.n_disc = self.n_bins + self.n_cats
        self.num_categorical_list = [2]*self.n_bins + self.cards
        
        if self.n_disc != 0:
            # Create a list to store individual embeddings
            self.embeddings_list = nn.ModuleList()
            
            # Create individual embeddings for each variable
            for num_categories in self.num_categorical_list:
                embedding = nn.Embedding(num_categories, emb_dim)
                self.embeddings_list.append(embedding)
        
        if self.n_nums != 0:
            self.mlp_nums = nn.Sequential(nn.Linear(16 * n_nums, 16 * n_nums),  # this should be 16 * n_nums, 16 * n_nums
                                          nn.SiLU(),
                                          nn.Linear(16 * n_nums, 16 * n_nums))
            
        self.mlp_output = nn.Sequential(nn.Linear(emb_dim * self.n_disc + 16 * n_nums, emb_dim),
                                       nn.ReLU(),
                                       nn.Linear(emb_dim, emb_dim))
        
    def forward(self, x):
        x_disc = x[:,:,0:self.n_disc].long().to(device)
        x_nums = x[:,:,self.n_disc:self.n_disc+self.n_nums].to(device)
        
        x_emb = torch.Tensor().to(device)
        
        # Binary + Discrete Variables
        if self.n_disc != 0:
            variable_embeddings = [embedding(x_disc[:,:,i]) for i, embedding in enumerate(self.embeddings_list)]
            x_disc_emb = torch.cat(variable_embeddings, dim=2)
            x_emb = x_disc_emb

        # Numerical Variables
        if self.n_nums != 0:
            x_nums = compute_sine_cosine(x_nums, num_terms=8)
            x_nums_emb = self.mlp_nums(x_nums)
            x_emb = torch.cat([x_emb, x_nums_emb], dim=2)
        
        final_emb = self.mlp_output(x_emb)
        
        return final_emb
    
################################################################################################################
def get_betas(steps):
    beta_start, beta_end = 1e-5, 0.1
    diffusion_ind = torch.linspace(0, 1, steps).to(device)
    return beta_start * (1 - diffusion_ind) + beta_end * diffusion_ind

diffusion_steps = 100
betas = get_betas(diffusion_steps)
alphas = torch.cumprod(1 - betas, dim=0)

def get_gp_covariance(t):
    s = t - t.transpose(-1, -2)
    diag = torch.eye(t.shape[-2]).to(t) 
    return diag 

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
    def __init__(self, input_size, org_feat_dim, hidden_size, num_layers, diffusion_steps, time_dim, emb_dim, n_bins, n_cats, n_nums, cards):
        super(BiRNN_score, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.input_proj = FeedForward(input_size, [], hidden_size)
        self.t_enc = PositionalEncoding(hidden_size, max_value=1)
        self.i_enc = PositionalEncoding(hidden_size, max_value=diffusion_steps) 
        self.proj = FeedForward(4 * hidden_size, [], hidden_size, final_activation=nn.ReLU())
        #self.proj = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.layer_norm = nn.LayerNorm(2 * hidden_size)
        self.fc = nn.Linear(2 * hidden_size, input_size)
        
        # self.mask_Emb = nn.Linear(org_feat_dim, hidden_size)
        self.Emb = Embedding_data(input_size, emb_dim, n_bins, n_cats, n_nums, cards)
        self.Emb_hidden = nn.Linear(emb_dim, hidden_size)
        self.cond_GRU = nn.GRU(emb_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.cond_output = nn.Linear(2*hidden_size, hidden_size)
        
        self.time_encode = nn.Sequential(nn.Linear(time_dim, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size))
        
    def forward(self, x, t, i, time_info = None, cond = None, target_mask = None):
        shape = x.shape

        x = x.view(-1, *shape[-2:])
        t = t.view(-1, shape[-2], 1)
        i = i.view(-1, shape[-2], 1)

        x = self.input_proj(x)
        t = self.t_enc(t)
        i = self.i_enc(i)

        if time_info is not None:
            time_info = self.time_encode(time_info)

        if cond is not None:
            Embedding = self.Emb(cond)
            if Embedding.shape[1] != shape[1]:
                padding_tensor = torch.zeros(shape[0], shape[1] - Embedding.shape[1], Embedding.shape[2]).to(device)
                Embedding = torch.cat((Embedding, padding_tensor), dim=1)
            cond_out, _ = self.cond_GRU(Embedding)
            x = self.proj(torch.cat([x + self.cond_output(cond_out), t, i, time_info], -1))    
        else:
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
def get_loss(model, x, t, time_info, cond = None, target_mask = None):
    i = torch.randint(0, diffusion_steps, size=(x.shape[0],))
    i = i.view(-1, 1, 1).expand_as(x[...,:1]).to(x)
    
    x_noisy, noise = add_noise(x, t, i)
    pred_noise = model(x_noisy, t, i, time_info, cond, target_mask)
    
    loss = (pred_noise - noise)**2
    return torch.mean(loss)

#####################################################################################################################
import copy
import tqdm.notebook
import random

def train_diffusion(latent_features, cond_df, data_dict, hidden_dim, num_layers, diffusion_steps, n_epochs, num_classes = None):
    
    emb_dim = 128
    
    parser = data_dict["parser"]
    datatype_info = parser.datatype_info()
    n_bins = datatype_info["n_bins"]
    n_cats = datatype_info["n_cats"]
    n_nums = datatype_info["n_nums"]
    cards  = datatype_info["cards"]
    
    cond_train=data_dict['cond_train']
    time_info=data_dict['time_info_train']
    target_mask=data_dict['target_mask_train']
    
    cond_tensor = cond_train.to(device) if cond_train is not None else None
    time_info = time_info.to(device)
    target_mask = target_mask.to(device)

    input_size = latent_features.shape[2]
    time_dim = time_info.shape[2]
    org_feat_dim = cond_train.shape[2]

    model = BiRNN_score(input_size, org_feat_dim, hidden_dim, num_layers, diffusion_steps, time_dim, emb_dim, n_bins, n_cats, n_nums, cards).to(device)
    optim = torch.optim.Adam(model.parameters())
    ema = EMA(beta=0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    x = latent_features.detach().to(device)
    N, T, K = latent_features.shape

    batch_size = diffusion_steps
    
    all_indices = list(range(len(latent_features)))
    
    with Progress() as progress:
        training_task = progress.add_task("[red]Training...", total=n_epochs)

        for epoch in range(n_epochs):
            batch_indices = random.sample(all_indices, batch_size)
            
            target_mask=data_dict['target_mask_train']
            cond_train=data_dict['cond_train']
            time_info=data_dict['time_info_train']
            
            target_mask = target_mask.float().to(device)
            cond_tensor = cond_train.to(device) if cond_train is not None else None
            time_info = time_info.to(device)
            
            optim.zero_grad()
            t = torch.rand(batch_size, T, 1).sort(1)[0].to(device)
            
            if cond_tensor is not None:
                loss = get_loss(model, x[batch_indices,:,:], t, time_info[batch_indices,:,:], cond_tensor[batch_indices,:,:], target_mask[batch_indices,:,:])
            else:
                loss = get_loss(model, x[batch_indices,:,:], t, time_info[batch_indices,:,:], None, None)

            loss.backward()
            optim.step()
            ema.step_ema(ema_model, model)
    
            progress.update(training_task, advance=1, description=f"Epoch {epoch}/{n_epochs} - Loss: {loss.item():.4f}")
    
    return model

#####################################################################################################################
@torch.no_grad()
def sample(t, B, T, F, diffusion_steps, model, time_info, cond, target_mask):
    x = torch.randn(B, T, F).to(device)
    
    time_info = time_info.to(device)
    cond = cond.to(device)

    for diff_step in reversed(range(0, diffusion_steps)):
        alpha = alphas[diff_step]
        beta = betas[diff_step]

        z = torch.randn(B, T, F).to(device)
        i = torch.Tensor([diff_step]).expand_as(x[...,:1]).to(device)
        predicted_noise = model(x, t, i, time_info, cond, target_mask)

        x = (1/(1 - beta).sqrt()) * (x - beta * predicted_noise / (1 - alpha).sqrt()) + beta.sqrt() * z

        if diff_step == 0:
            x = (1/(1 - beta).sqrt()) * (x - beta * predicted_noise / (1 - alpha).sqrt())
    return x