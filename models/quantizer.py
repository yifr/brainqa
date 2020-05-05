import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import logging

log = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline: 
            1. Flatten input 
            2. Get embedding distance
            3. Embed input 
            4. Compute loss
        """
        z = z.permute(0, 2, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # log.info("Z-Flattened shape: {}".format(z_flattened.shape))

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # log.info('Distance calculation shape: {}, example: {}'.format(d.shape, d[0]))

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 2, 1).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices


class VectorQuantizerRandomRestart(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE. 

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - lambda : decay parameter 

    This Quantizer solves the issue of index collapse
    using a random restart strategy for the discrete latent space.
    If the mean usage of a latent index falls below a threshold  
    it is randomly reassigned to an encoder output.   
    """

    def __init__(self, n_e, e_dim, beta, restart_threshold=1.0):
        super(VectorQuantizerRandomRestart, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.restart_threshold = restart_threshold
        self.k_sum = self.embedding.weight.data.to(self.device)
        self.k_elem = torch.ones(self.n_e, device=self.device)

    def _tile(self, x):
        d, ew = x.shape
        if d < self.n_e:
            n_repeats = (self.n_e + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def update_embed_idx(self, z_enc, z_quant_onehot, mu=.8):
        with torch.no_grad():
            _k_elem = z_quant_onehot.sum(dim=0)
            #log.info('z_quant shape: {} \tz_enc shape: {}'.format(z_quant_onehot.shape, z_enc.shape))
            _k_sum = torch.matmul(z_quant_onehot.t(), z_enc)
            y = self._tile(z_enc)
            _k_rand = y[torch.randperm(y.shape[0])][:self.n_e]

            #log.info('k_elem shape: {} \t_k_elem shape: {}'.format(self.k_elem.shape, _k_elem.shape))
            self.k_sum = mu * self.k_sum + (1. - mu) * _k_sum
            self.k_elem = mu * self.k_elem + (1. - mu) * _k_elem

            usage = (self.k_elem.view(self.n_e, 1) >= self.restart_threshold).float()

            self.embedding.weight.data = usage * (self.k_sum.view(self.n_e, self.e_dim) / self.k_elem.view(self.n_e, 1)) \
                        + (1 - usage) * _k_rand
        

    def forward(self, z):
        """
        Same as regular VectorQuantization but with an update check
        """
        #CHANGING PERMUTES z = z.permute(0, 2, 3, 1).contiguous()
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # log.info("Z-Flattened shape: {}".format(z_flattened.shape))

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # log.info('Distance calculation shape: {}, example: {}'.format(d.shape, d[0]))

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        #log.info('Z_flatten shape: {} \t Min_enc shape: {} \t Z shape: {}'. format(z_flattened.shape, min_encodings.shape, z.shape))
        #update embeddings if necessary (random restart)
        self.update_embed_idx(z_flattened.float(), min_encodings.float())

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 2, 1).contiguous()

        

        return loss, z_q, perplexity, min_encodings, min_encoding_indices
