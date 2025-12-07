import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import tqdm.notebook
import gc
import random
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import numpy as np
import process_edited as pce
from torch.optim import Adam
import math
from rich.progress import Progress
import dataprovider as dp
import os

torch.cuda.empty_cache()
#torch.backends.cuda.preferred_linalg_library('magma')   # or 'cusolver'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # This is your global device

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
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.RNN = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, d_last_states = self.RNN(x)
        y_hat_logit = self.fc(d_last_states[-1])
        y_hat = torch.sigmoid(y_hat_logit)
        return y_hat

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
            
        self.mlp_output = nn.Sequential(nn.Linear(emb_dim * self.n_disc + 16 * n_nums, emb_dim), # this should be 16 * n_nums, 16 * n_nums
                                       nn.ReLU(),
                                       nn.Linear(emb_dim, input_size))
        
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
class DeapStack(nn.Module):
    def __init__(self, channels, n_bins, n_cats, n_nums, cards, input_size, hidden_size, num_layers, cat_emb_dim, time_dim, lat_dim):
        super().__init__()
        self.Emb = Embedding_data(input_size, cat_emb_dim, n_bins, n_cats, n_nums, cards)
        self.time_encode = nn.Sequential(nn.Linear(time_dim, input_size),
                                         nn.ReLU(),
                                         nn.Linear(input_size, input_size))
        
        self.encoder_mu = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.encoder_logvar = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
        self.fc_mu = nn.Linear(hidden_size, lat_dim)
        self.fc_logvar = nn.Linear(hidden_size, lat_dim)
        
        #self.decoder_proj_in = nn.Linear(lat_dim, hidden_size)
        #self.decoder_mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)

        self.decoder_mlp = nn.Sequential(nn.Linear(lat_dim, hidden_size),
                                         nn.ReLU(),
                                         nn.Linear(hidden_size, hidden_size))
        
        self.Emb_decoder = Embedding_data(input_size, cat_emb_dim, n_bins, n_cats, n_nums, cards)
        self.Emb_hidden_decoder = nn.Linear(input_size, hidden_size)

        self.channels = channels
        self.n_bins = n_bins
        self.n_cats = n_cats
        self.n_nums = n_nums
        self.cards = cards
        self.disc = self.n_bins + self.n_cats
        self.sigmoid = torch.nn.Sigmoid ()

        self.bins_linear = nn.Linear(hidden_size, n_bins) if n_bins else None
        self.cats_linears = nn.ModuleList([nn.Linear(hidden_size, card) for card in cards]) if n_cats else None 
        self.nums_linear = nn.Linear(hidden_size, n_nums) if n_nums else None

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encoder(self, x):
        x = self.Emb(x)
        mu_z, _ = self.encoder_mu(x)
        logvar_z, _ = self.encoder_logvar(x)
        
        mu_z = self.fc_mu(mu_z); logvar_z = self.fc_logvar(logvar_z)
        emb = self.reparametrize(mu_z, logvar_z)
        
        return emb, mu_z, logvar_z

    def decompose_target_mask(self, mask):
        B, L, _ = mask.shape

        ###### Binary data type mask
        if self.n_bins is None or self.n_bins == 0:
            mask_bin = torch.zeros((B, L, 0), dtype=torch.float32).to(device)  # Return zero matrix
        else:
            mask_bin_temp = mask[:, :, :self.n_bins].to(device)
            mask_bin = torch.zeros_like(mask_bin_temp, dtype=torch.float32).to(device)
            mask_bin[mask_bin_temp == 0] = float('-100')

        ###### Categorical data type mask
        if self.n_cats is None or self.n_cats == 0:
            mask_cat = [torch.zeros((B, L, 0), dtype=torch.float32).to(device)]  # Return a zero list
        else:
            mask_cat_temp = mask[:, :, self.n_bins:self.n_bins + self.n_cats].to(device)
            mask_cat = []
            for i in range(self.n_cats):
                mask_cat.append(torch.zeros(B * L, self.cards[i]).to(device))
                missing_indices = (mask_cat_temp[:, :, i].reshape(B * L) == 0).to(device)
                mask_cat[i][missing_indices, 0] = float('100')
                mask_cat[i] = mask_cat[i].reshape(B, L, -1)

        ###### Numerical data type mask
        if self.n_nums is None or self.n_nums == 0:
            mask_num = torch.zeros((B, L, 0), dtype=torch.float32).to(device)  # Return zero matrix
        else:
            mask_num = mask[:, :, self.disc:self.disc+self.n_nums].to(device)

        return mask_bin, mask_cat, mask_num

    def decoder(self, latent_feature, mask = None, cond=None):
        decoded_outputs = dict()
        latent_feature = self.decoder_mlp(latent_feature)
        
        #latent_proj = self.decoder_proj_in(latent_feature)
        #latent_feature, _ = self.decoder_mha(latent_proj, latent_proj, latent_proj)
        
        mask_bin, mask_cat, mask_num = self.decompose_target_mask(mask)

        if cond is not None:
            cond_embedding = self.Emb_decoder(cond)
            latent_feature += self.Emb_hidden_decoder(cond_embedding)

        if self.bins_linear:
            decoded_outputs['bins'] = self.bins_linear(latent_feature) + mask_bin

        if self.cats_linears:
            decoded_outputs['cats'] = [linear(latent_feature) + mask_cat_el for linear, mask_cat_el in zip(self.cats_linears, mask_cat)]

        if self.nums_linear:
            decoded_outputs['nums'] = self.sigmoid(self.nums_linear(latent_feature)) * mask_num

        return decoded_outputs

    def forward(self, x, mask, cond=None):
        emb, mu_z, logvar_z = self.encoder(x)
        outputs = self.decoder(emb, mask, cond)
        return outputs, emb, mu_z, logvar_z

def auto_loss(inputs, reconstruction, n_bins, n_nums, n_cats, cards):

    """ Calculating the loss for DAE network.
        BCE for masks and reconstruction of binary inputs.
        CE for categoricals.
        MSE for numericals.
        reconstruction loss is weighted average of mean reduction of loss per datatype.
        mask loss is mean reduced.
        final loss is weighted sum of reconstruction loss and mask loss.
    """
    B, L, _ = inputs.shape

    bins = inputs[:,:,0:n_bins]
    cats = inputs[:,:,n_bins:n_bins+n_cats].long()
    nums = inputs[:,:,n_bins+n_cats:n_bins+n_cats+n_nums]

    disc_loss = 0; num_loss = 0
    
    if 'bins' in reconstruction:
        disc_loss += F.binary_cross_entropy_with_logits(reconstruction['bins'], bins)

    if 'cats' in reconstruction:
        cats_losses = []
        for i in range(len(reconstruction['cats'])):
            cats_losses.append(F.cross_entropy(reconstruction['cats'][i].reshape(B*L, cards[i]), \
                                               cats[:,:,i].unsqueeze(2).reshape(B*L, 1).squeeze(1)))
        disc_loss += torch.stack(cats_losses).mean()

    if 'nums' in reconstruction:
        num_loss = F.mse_loss(reconstruction['nums'], nums)

    return disc_loss, num_loss

import copy
import random
import torch
from torch.optim import Adam
from rich.progress import Progress

def train_autoencoder(
        real_df, channels, hidden_size, num_layers, lr, weight_decay, n_epochs,
        batch_size, threshold, min_beta, max_beta, emb_dim, time_dim, lat_dim,
        device, data_dict):
    """
    Train an auto‑encoder and select the weights that minimize validation loss.
    Returns
    -------
    best_ae : nn.Module
        Model after restoring the best (lowest‑val‑loss) weights.
    latent_features : torch.Tensor
        Latent features for the **training set** under the best model.
    output : torch.Tensor
        Reconstruction of the **training set** under the best model.
    history : dict
        Dict with keys 'train', 'val' listing per‑epoch total losses.
    """
    # ---------- preprocessing ----------
    parser = data_dict["parser"]
    info   = parser.datatype_info()
    n_bins, n_cats, n_nums, cards = (
        info['n_bins'],
        info['n_cats'],
        info['n_nums'],
        info['cards'],
    )
    
    target_train = data_dict['response_train']
    N, _, input_size = target_train.shape

    ae = DeapStack(channels, n_bins, n_cats, n_nums, cards,
                   input_size, hidden_size, num_layers,
                   emb_dim, time_dim, lat_dim).to(device)

    optimizer = Adam(ae.parameters(), lr=lr, weight_decay=weight_decay)

    # ---------- validation tensors ----------
    val_resp   = data_dict['response_val'].to(device)
    val_mask   = data_dict['target_mask_val'].to(device)
    val_cond   = data_dict['cond_val']
    val_cond   = val_cond.to(device) if val_cond is not None else None
    if val_cond is not None:
        val_inp = val_resp + val_cond
    else:
        val_inp = val_resp
    val_target = val_inp * val_mask      # ground‑truth values for loss calc

    # ---------- bookkeeping ----------
    beta               = max_beta
    patience, lambd    = 0, 0.7
    best_val_loss      = float('inf')
    best_state         = copy.deepcopy(ae.state_dict())
    history            = {'train': [], 'val': []}

    all_indices        = list(range(N))

    with Progress() as progress:
        task = progress.add_task("[red]Training", total=n_epochs)

        for epoch in range(n_epochs):
            # ---- training minibatch ----
            ae.train()
            batch_idx = random.sample(all_indices, batch_size)

            # fresh view each epoch in case dict changed
            tgt_train  = data_dict['response_train'].to(device)
            mask_train = data_dict['target_mask_train'].to(device)
            cond_train = data_dict['cond_train']
            cond_train = cond_train.to(device) if cond_train is not None else None

            if cond_train is not None:
                inp = tgt_train + cond_train
            else:
                inp = tgt_train

            target = inp * mask_train

            optimizer.zero_grad()
            if cond_train is not None:
                out, _, mu, logvar = ae(inp[batch_idx],
                                        mask_train[batch_idx],
                                        cond_train[batch_idx])
            else:
                out, _, mu, logvar = ae(inp[batch_idx],
                                        mask_train[batch_idx],
                                        None)

            disc_loss, num_loss = auto_loss(target[batch_idx], out,
                                            n_bins, n_nums, n_cats, cards)
            kld = -0.5 * torch.mean((1 + logvar - mu.pow(2) - logvar.exp())
                                    .mean(-1).mean())
            train_loss = num_loss + disc_loss + beta * kld
            train_loss.backward()
            optimizer.step()

            # ---- validation (no grad, full set) ----
            ae.eval()
            with torch.no_grad():
                if val_cond is not None:
                    v_out, _, v_mu, v_logvar = ae(val_inp, val_mask, val_cond)
                else:
                    v_out, _, v_mu, v_logvar = ae(val_inp, val_mask, None)

                v_disc, v_num = auto_loss(val_target, v_out,
                                          n_bins, n_nums, n_cats, cards)
                v_kld = -0.5 * torch.mean((1 + v_logvar - v_mu.pow(2) - v_logvar.exp())
                                          .mean(-1).mean())
                val_loss = v_num + v_disc + beta * v_kld

            # ---- early‑stopping / beta anneal ----
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = copy.deepcopy(ae.state_dict())
                patience      = 0
            else:
                patience += 1
                if patience == 10 and beta > min_beta:
                    beta *= lambd
                    patience = 0   # reset counter after annealing

            # ---- history & progress bar ----
            history['train'].append(train_loss.item())
            history['val'].append(val_loss.item())
            progress.update(task, advance=1,
                            description=(f"Epoch {epoch+1}/{n_epochs}  "
                                         f"Train: {train_loss.item():.4f}  "
                                         f"Val: {val_loss.item():.4f}  "
                                         f"β={beta:.4f}"))

    # ---------- restore best weights ----------
    ae.load_state_dict(best_state)
    ae.eval()

    # latent features & reconstructions for the *training* set
    full_cond = data_dict['cond_train']
    full_cond = full_cond.to(device) if full_cond is not None else None
    full_mask = data_dict['target_mask_train'].to(device)
    full_inp  = data_dict['response_train'].to(device)
    if full_cond is not None:
        full_inp = full_inp + full_cond

    output, latent, mu_z, logvar_z = ae(full_inp, full_mask, full_cond)

    return ae, latent.detach(), output, history


import torch
import torch.nn as nn
from torch.optim import Adam
import copy
import random
from rich.progress import Progress

# (DeapStack, auto_loss 등 필요한 다른 import 및 클래스/함수 정의가 있다고 가정)

def train_autoencoder_best_train(
        real_df, channels, hidden_size, num_layers, lr, weight_decay, n_epochs,
        batch_size, threshold, min_beta, max_beta, emb_dim, time_dim, lat_dim,
        device, data_dict):
    """
    Train an auto-encoder and select the weights that minimize *training* loss.
    
    Returns
    -------
    best_ae : nn.Module
        Model after restoring the best (lowest-train-loss) weights.
    latent_features : torch.Tensor
        Latent features for the **training set** under the best model.
    output : torch.Tensor
        Reconstruction of the **training set** under the best model.
    history : dict
        Dict with keys 'train', 'val' listing per-epoch total losses.
    """
    # ---------- preprocessing ----------
    parser = data_dict["parser"]
    info   = parser.datatype_info()
    n_bins, n_cats, n_nums, cards = (
        info['n_bins'],
        info['n_cats'],
        info['n_nums'],
        info['cards'],
    )
    
    target_train = data_dict['response_train']
    N, _, input_size = target_train.shape

    ae = DeapStack(channels, n_bins, n_cats, n_nums, cards,
                   input_size, hidden_size, num_layers,
                   emb_dim, time_dim, lat_dim).to(device)

    optimizer = Adam(ae.parameters(), lr=lr, weight_decay=weight_decay)

    # ---------- validation tensors ----------
    val_resp   = data_dict['response_val'].to(device)
    val_mask   = data_dict['target_mask_val'].to(device)
    val_cond   = data_dict['cond_val']
    val_cond   = val_cond.to(device) if val_cond is not None else None
    if val_cond is not None:
        val_inp = val_resp + val_cond
    else:
        val_inp = val_resp
    val_target = val_inp * val_mask      # ground-truth values for loss calc

    # ---------- bookkeeping ----------
    beta              = max_beta
    patience, lambd   = 0, 0.7
    best_val_loss     = float('inf')
    best_train_loss   = float('inf')  # <--- [추가] train_loss 추적용
    best_state        = copy.deepcopy(ae.state_dict())
    history           = {'train': [], 'val': []}

    all_indices       = list(range(N))

    with Progress() as progress:
        task = progress.add_task("[red]Training", total=n_epochs)

        for epoch in range(n_epochs):
            # ---- training minibatch ----
            ae.train()
            batch_idx = random.sample(all_indices, batch_size)

            # fresh view each epoch in case dict changed
            tgt_train  = data_dict['response_train'].to(device)
            mask_train = data_dict['target_mask_train'].to(device)
            cond_train = data_dict['cond_train']
            cond_train = cond_train.to(device) if cond_train is not None else None

            if cond_train is not None:
                inp = tgt_train + cond_train
            else:
                inp = tgt_train

            target = inp * mask_train

            optimizer.zero_grad()
            if cond_train is not None:
                out, _, mu, logvar = ae(inp[batch_idx],
                                        mask_train[batch_idx],
                                        cond_train[batch_idx])
            else:
                out, _, mu, logvar = ae(inp[batch_idx],
                                        mask_train[batch_idx],
                                        None)

            disc_loss, num_loss = auto_loss(target[batch_idx], out,
                                            n_bins, n_nums, n_cats, cards)
            kld = -0.5 * torch.mean((1 + logvar - mu.pow(2) - logvar.exp())
                                    .mean(-1).mean())
            train_loss = num_loss + disc_loss + beta * kld
            train_loss.backward()
            optimizer.step()

            # ---- [수정] train_loss 기준으로 best_state 저장 ----
            current_train_loss = train_loss.item()
            if current_train_loss < best_train_loss:
                best_train_loss = current_train_loss
                best_state = copy.deepcopy(ae.state_dict())
            # ----------------------------------------------------

            # ---- validation (no grad, full set) ----
            ae.eval()
            with torch.no_grad():
                if val_cond is not None:
                    v_out, _, v_mu, v_logvar = ae(val_inp, val_mask, val_cond)
                else:
                    v_out, _, v_mu, v_logvar = ae(val_inp, val_mask, None)

                v_disc, v_num = auto_loss(val_target, v_out,
                                          n_bins, n_nums, n_cats, cards)
                v_kld = -0.5 * torch.mean((1 + v_logvar - v_mu.pow(2) - v_logvar.exp())
                                          .mean(-1).mean())
                val_loss = v_num + v_disc + beta * v_kld

            # ---- [수정] early-stopping / beta anneal (best_state 저장 로직 제거) ----
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # best_state  = copy.deepcopy(ae.state_dict()) # <--- [삭제]
                patience      = 0
            else:
                patience += 1
                if patience == 10 and beta > min_beta:
                    beta *= lambd
                    patience = 0   # reset counter after annealing

            # ---- history & progress bar ----
            history['train'].append(current_train_loss) # .item()을 위에서 처리
            history['val'].append(val_loss.item())
            progress.update(task, advance=1,
                            description=(f"Epoch {epoch+1}/{n_epochs}  "
                                         f"Train: {current_train_loss:.4f}  "
                                         f"Val: {val_loss.item():.4f}  "
                                         f"β={beta:.4f}"))

    # ---------- restore best weights ----------
    ae.load_state_dict(best_state)
    ae.eval()

    # latent features & reconstructions for the *training* set
    full_cond = data_dict['cond_train']
    full_cond = full_cond.to(device) if full_cond is not None else None
    full_mask = data_dict['target_mask_train'].to(device)
    full_inp  = data_dict['response_train'].to(device)
    if full_cond is not None:
        full_inp = full_inp + full_cond

    output, latent, mu_z, logvar_z = ae(full_inp, full_mask, full_cond)

    return ae, latent.detach(), output, history
