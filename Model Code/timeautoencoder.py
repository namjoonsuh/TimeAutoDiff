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
import DP as dp
import math
from rich.progress import Progress

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    sine_values = sine_values.view(*sine_values.shape[:-2], -1)
    cosine_values = cosine_values.view(*cosine_values.shape[:-2], -1)

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
#def get_torch_trans(heads = 8, layers = 1, channels = 64):
#    encoder_layer = nn.TransformerEncoderLayer(d_model = channels, nhead = heads, dim_feedforward=64, activation = "gelu")
#    return nn.TransformerEncoder(encoder_layer, num_layers = layers)

#class Transformer_Block(nn.Module):
#    def __init__(self, channels):
#        super().__init__()
#        self.channels = channels
        
#        self.conv_layer1 = nn.Conv1d(1, self.channels, 1)
#        self.feature_layer = get_torch_trans(heads = 8, layers = 1, channels = self.channels)
#        self.conv_layer2 = nn.Conv1d(self.channels, 1, 1)
    
#    def forward_feature(self, y, base_shape):
#        B, channels, L, K = base_shape
#        if K == 1:
#            return y.squeeze(1)
#        y = y.reshape(B, channels, L, K).permute(0, 2, 1, 3).reshape(B*L, channels, K)
#        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
#        y = y.reshape(B, L, channels, K).permute(0, 2, 1, 3)
#        return y
    
#    def forward(self, x):
#        x = x.unsqueeze(1)
#        B, input_channel, K, L = x.shape
#        base_shape = x.shape

#        x = x.reshape(B, input_channel, K*L)       
        
#        conv_x = self.conv_layer1(x).reshape(B, self.channels, K, L)
#        x = self.forward_feature(conv_x, conv_x.shape)
#        x = self.conv_layer2(x.reshape(B, self.channels, K*L)).squeeze(1).reshape(B, K, L)
        
#        return x

################################################################################################################
class DeapStack(nn.Module):
    def __init__(self, channels, batch_size, seq_len, n_bins, n_cats, n_nums, cards, input_size, hidden_size, num_layers, cat_emb_dim, time_dim, lat_dim):
        super().__init__()
        self.Emb = Embedding_data(input_size, cat_emb_dim, n_bins, n_cats, n_nums, cards)
        self.time_encode = nn.Sequential(nn.Linear(time_dim, input_size),
                                         nn.ReLU(),
                                         nn.Linear(input_size, input_size))
        #self.encoder_Transformer = Transformer_Block(channels)
        
        self.encoder_mu = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.encoder_logvar = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
        self.fc_mu = nn.Linear(hidden_size, lat_dim)
        self.fc_logvar = nn.Linear(hidden_size, lat_dim)

        #self.cont_normed = nn.LayerNorm((seq_len, n_nums))
        #self.decoder_Transformer = Transformer_Block(channels)
        #self.decoder_rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)

        self.decoder_mlp = nn.Sequential(nn.Linear(lat_dim, hidden_size),
                                         nn.ReLU(),
                                         nn.Linear(hidden_size, hidden_size))
        
        self.channels = channels
        self.n_bins = n_bins
        self.n_cats = n_cats
        self.n_nums = n_nums
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
        #x = self.encoder_Transformer(x)
        #x = x + self.time_encode(time_info)
        
        mu_z, _ = self.encoder_mu(x)
        logvar_z, _ = self.encoder_logvar(x)
        
        mu_z = self.fc_mu(mu_z); logvar_z = self.fc_logvar(logvar_z)
        emb = self.reparametrize(mu_z, logvar_z)
        
        return emb, mu_z, logvar_z

    def decoder(self, latent_feature):
        decoded_outputs = dict()
        latent_feature = self.decoder_mlp(latent_feature)
        
        B, L, K = latent_feature.shape
        
        if self.bins_linear:
            decoded_outputs['bins'] = self.bins_linear(latent_feature)

        if self.cats_linears:
            decoded_outputs['cats'] = [linear(latent_feature) for linear in self.cats_linears]

        if self.nums_linear:
            decoded_outputs['nums'] = self.sigmoid(self.nums_linear(latent_feature))

        return decoded_outputs

    def forward(self, x):
        emb, mu_z, logvar_z = self.encoder(x)
        outputs = self.decoder(emb)
        return outputs, emb, mu_z, logvar_z
    
def auto_loss(inputs, reconstruction, n_bins, n_nums, n_cats, beta, cards):
    """ Calculating the loss for DAE network.
        BCE for masks and reconstruction of binary inputs.
        CE for categoricals.
        MSE for numericals.
        reconstruction loss is weighted average of mean reduction of loss per datatype.
        mask loss is mean reduced.
        final loss is weighted sum of reconstruction loss and mask loss.
    """
    B, L, K = inputs.shape

    bins = inputs[:,:,0:n_bins]
    cats = inputs[:,:,n_bins:n_bins+n_cats].long()
    nums = inputs[:,:,n_bins+n_cats:n_bins+n_cats+n_nums]

    #reconstruction_losses = dict()
    disc_loss = 0; num_loss = 0;
    
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

    #reconstruction_loss = torch.stack(list(reconstruction_losses.values())).mean()

    return disc_loss, num_loss

def train_autoencoder(real_df, processed_data, channels, hidden_size, num_layers, lr, weight_decay, n_epochs, batch_size, threshold, min_beta, max_beta, emb_dim, time_dim, lat_dim, device):

    parser = pce.DataFrameParser().fit(real_df, threshold)
    data = parser.transform()
    data = torch.tensor(data.astype('float32')).unsqueeze(0)
        
    datatype_info = parser.datatype_info()
    n_bins = datatype_info['n_bins']; n_cats = datatype_info['n_cats']
    n_nums = datatype_info['n_nums']; cards = datatype_info['cards']
    
    N, seq_len, input_size = processed_data.shape
    ae = DeapStack(channels, batch_size, seq_len, n_bins, n_cats, n_nums, cards, input_size, hidden_size, num_layers, emb_dim, time_dim, lat_dim).to(device)
    
    optimizer_ae = Adam(ae.parameters(), lr=lr, weight_decay=weight_decay)

    inputs = processed_data.to(device)
        
    losses = []
    recons_loss = []
    KL_loss = []
    beta = max_beta
    
    lambd = 0.7
    best_train_loss = float('inf')
    all_indices = list(range(N))
    
    with Progress() as progress:
        training_task = progress.add_task("[red]Training...", total=n_epochs)

        for epoch in range(n_epochs):
            ######################### Train Auto-Encoder #########################
            batch_indices = random.sample(all_indices, batch_size)
    
            optimizer_ae.zero_grad()
            outputs, _, mu_z, logvar_z = ae(inputs[batch_indices,:,:])
            
            disc_loss, num_loss = auto_loss(inputs[batch_indices,:,:], outputs, n_bins, n_nums, n_cats, beta, cards)
            temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
            loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
            
            loss_Auto = num_loss + disc_loss + beta * loss_kld
            loss_Auto.backward()
            optimizer_ae.step()
            progress.update(training_task, advance=1, description=f"Epoch {epoch}/{n_epochs} - Loss: {loss_Auto.item():.4f}")
            
            if loss_Auto < best_train_loss:
                best_train_loss = loss_Auto
                patience = 0
            else:
                patience += 1
                if patience == 10:
                    if beta > min_beta:
                        beta = beta * lambd
            
            #recons_loss.append(num_loss.item() + disc_loss.item())
            #KL_loss.append(loss_kld.item())
    
    output, latent_features, _, _ = ae(processed_data)
        
    #return (ae, latent_features.detach(), output, losses, recons_loss, KL_loss)
    return (ae, latent_features.detach(), output, losses, recons_loss, mu_z, logvar_z)

