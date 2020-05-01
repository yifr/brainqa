import torch
import torch.nn as nn
import numpy as np
from models.quantizer import VectorQuantizer
from models.decoder import Decoder
from models.encoder import Encoder

import logging

log = logging.getLogger(__name__)

#Need to change to word embeddings
class VQVAE(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, save_img_embedding_map=False):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(in_dim=256, h_dim=h_dim, n_res_layers=n_res_layers, res_h_dim=res_h_dim)
        self.pre_quantization_conv = nn.Conv1d(
            h_dim, embedding_dim, kernel_size=3, stride=1, padding=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        #E_indices used in sampling, just save last to rep last latent state
        self.e_indices = None

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False):
        #log.info('Embeddings shape: {}'.format(x.shape))
        z_e = self.encoder(x)
        #log.info('Z-e shape before pre_conv: {}'.format(z_e.shape))
        z_e = self.pre_quantization_conv(z_e)
        #log.info('Z-e shape: {}'.format(z_e.shape))
        embedding_loss, z_q, perplexity, _, e_indices = self.vector_quantization(z_e)
        #log.info('Z-q shape: {}'.format(z_q.shape))
        #Retain the embedding indices to be used in sampling
        self.e_indices = e_indices

        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert x.shape == x_hat.shape

        #log.info('Recon embed shape: {}'.format(x_hat.shape))
        return embedding_loss, x_hat, perplexity, z_q, e_indices
