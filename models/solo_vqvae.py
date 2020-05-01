import torch
import torch.nn as nn
import numpy as np
from models.quantizer import VectorQuantizer
from models.decoder import Decoder
from models.encoder import Encoder
from transformers import BertModel, BertTokenizer, BertPreTrainedModel, BertConfig

import logging

log = logging.getLogger(__name__)

#Need to change to word embeddings
class S_VQVAE(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, save_img_embedding_map=False):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        #self.encoder = Encoder(in_dim=256, h_dim=h_dim, n_res_layers=n_res_layers, res_h_dim=res_h_dim)
        self.pre_quantization_conv = nn.Conv1d(
            h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        #E_indices used in sampling, just save last to rep last latent state
        self.e_indices = None

        self.bert = BertModel.from_pretrained(args.model_name_or_path)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        verbose=False):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #B = Batch Size, S = Sequence Length, H = Hidden Size
        #outputs_encoder = (last_hidden_state: (BxSxH), pooler_output:(BxH), hidden_states: (BxSxH))
        bert_embeds = self.bert.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        #z_e = self.encoder(bert_embeds)

        z_e = self.pre_quantization_conv(bert_embeds)
        embedding_loss, z_q, perplexity, _, e_indices = self.vector_quantization(z_e)

        #Retain the embedding indices to be used in sampling
        self.e_indices = e_indices

        x_hat = self.decoder(z_q)
        

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert x.shape == x_hat.shape

        bert_outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=bert_embeds
            )
        last_hidden_state_vqvae, pooler_output, hidden_states = bert_outputs

        # Compute logits 
        logits = self.qa_outputs(last_hidden_state_vqvae)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) #+ outputs_encoder_vqvae[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2 
            
            # Compute VQVAE loss
            vq_recon_loss = torch.mean((embeds_reconstructed - bert_embeds)**2) # VQVAE divides this by variance of total training data 
            vqvae_loss = vq_recon_loss + vq_embedding_loss       
                        
            outputs = (total_loss,vqvae_loss) + outputs

        return embedding_loss, x_hat, perplexity, z_q
