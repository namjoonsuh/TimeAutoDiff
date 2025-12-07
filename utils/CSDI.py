#@title Code for Score network with Transformer-layer (2D) - V2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 6  21:46:00 2023
@author: Namjoon Suh
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def time_embedding(pos, d_model = 128):
    # This function takes input S = torch.range(L) and gives output of the form (1, L, time_dim)
    pos = pos.unsqueeze(0)
    pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
    position = pos.unsqueeze(2)
    div_term = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2) / d_model)
    return torch.cat([torch.sin(position * div_term), torch.cos(position * div_term)], dim=-1)

def side_info(S, B, K, L):
    time_embed = time_embedding(S).unsqueeze(2).expand(B, -1, K, -1)  # (B, L, K, time_embd_size)

    #feature_embed_layer = nn.Embedding(num_embeddings = K, embedding_dim = 16)
    #feature_embed = feature_embed_layer(torch.arange(K)) # (target_dim = K, feat_embedding_dim = 16)
    #feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1) # (B, L, K, feat_embedding_dim)

    side_info = time_embed
    #side_info = torch.cat([time_embed , feature_embed], dim=-1) # (B, L, K, time+feat_embedding_dim)
    side_info = side_info.permute(0, 3, 2, 1)  # output is of shape (B, feat_embd_size, K, L)

    return side_info

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(conv_layer.weight)
    return conv_layer

def get_torch_trans(heads = 8, layers = 1, channels = 64):
    encoder_layer = nn.TransformerEncoderLayer(d_model = channels, nhead = heads, dim_feedforward=64, activation = "gelu")
    return nn.TransformerEncoder(encoder_layer, num_layers = layers)

class DiffusionEmbedding(nn.Module):
  """Gaussian random features for encoding time steps."""
  def __init__(self, embedding_size, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embedding_size//2) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = 2 * np.pi * x[:, None] * self.W[None, :]
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffsuion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.temporal_module = []
        for k in range(8):
            self.temporal_module.append( get_torch_trans(heads = nheads, layers = 1, channels = channels)  )
        self.feature_layer = get_torch_trans(heads = nheads, layers = 1, channels = channels)

    # Apply 1-layer transformerEncoder along the time-axis.
    # Take (B, channel, K*L) and output (B, channel, K*L)
    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        temporal_list = []
        y = y.reshape(B, channel, K, L)
        split_tensors = torch.split(y, 1, dim = 2)                                # Split into K tensors of shape (B, C, 1, L)

        for k in range(K):
            y = split_tensors[k].squeeze(2).permute(2, 0, 1)                      # (B, C, 1, L) -> (B, C, L) -> (L, B, C)
            temp = self.temporal_module[k].to(device)                             # No batch-first option, src = (sequence_len, batch_size, embd_dim)
            y = temp(y).permute(1, 2, 0).unsqueeze(2)                             # (L, B, C) -> (B, C, L) -> (B, C, 1, L)
            temporal_list.append(y)                                               # Concat K-(B, C, 1, L) tensors

        y = torch.cat(temporal_list, dim = 2).reshape(B, channel, K*L)            # (B, C, K, L) -> (B, C, K*L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B*L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 1, 3).reshape(B, channel, K*L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape

        x = x.reshape(B, channel, K*L)
        diffusion_emb = self.diffsuion_projection(diffusion_emb).unsqueeze(-1)

        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        #y = self.forward_feature(y, base_shape)     # (B, channel, K*L)
        y = self.mid_projection(y)                   # (B, 2*channel, K*L)

        _, cond_dim, _, _ = cond_info.shape          # cond_info = (B, side_dim, K, L)
        cond_info = cond_info.reshape(B, cond_dim, K*L)
        cond_info = self.cond_projection(cond_info)  # (B, 2*channel, K*L)
        y = y + cond_info

        gate, filt = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip # (B, channel, K, L)

class ScoreNetTrans_v2(nn.Module):
    def __init__(self, inputdim, channels, diffusion_embedding_dim, side_dim, nheads, res_layers):
        super().__init__()

        self.inputdim = inputdim
        self.channels = channels
        self.diffusion_embedding_dim = diffusion_embedding_dim
        self.side_dim = side_dim
        self.nheads = nheads
        self.res_layers = res_layers
        self.DiffusionEmbedding = DiffusionEmbedding(diffusion_embedding_dim)

        self.input_projection = Conv1d_with_init(self.inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim = self.side_dim,
                    channels = self.channels,
                    diffusion_embedding_dim = self.diffusion_embedding_dim,
                    nheads = self.nheads,
                )
                for _ in range(self.res_layers)
            ]
        )

    def forward(self, x, cond_info, diffusion_step):
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K*L)
        x = self.input_projection(x)    #Increase inputdim to output_channel_dim
        x = F.relu(x)

        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.DiffusionEmbedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers)) # (B, channel, K, L)
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)      # (B, channel, K*L)
        x = F.relu(x)
        x = self.output_projection2(x)      # (B, channel, K*L)
        x = x.reshape(B, K, L)
        return x