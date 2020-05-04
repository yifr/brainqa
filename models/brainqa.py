from transformers import BertModel, BertTokenizer, BertPreTrainedModel, BertConfig
from models.vqvae import VQVAE

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np

import logging


log = logging.getLogger(__name__)

class BrainQA(BertPreTrainedModel):
    def __init__(self, args, config):
        super(BrainQA, self).__init__(config)
        self.num_labels = config.num_labels

        # Set up BERT encoder
        self.config_enc = config.to_dict()
        self.config_enc['output_hidden_states'] = True
        self.config_enc = BertConfig.from_dict(self.config_enc)
        self.bert_enc = BertModel.from_pretrained(args.model_name_or_path, config=self.config_enc)

        # Set up BERT decoder
        # self.config_dec = config.to_dict()
        # self.config_dec['is_decoder'] = True
        # self.config_dec = BertConfig.from_dict(self.config_dec)
        # self.bert_dec = BertModel(self.config_dec)

        # # VQVAE for external memory
        n_embeddings = 4096
        self.vqvae_model= VQVAE(h_dim=256,
                        res_h_dim=256,
                        n_res_layers=4,
                        n_embeddings=n_embeddings,
                        embedding_dim=256,
                        restart=True,
                        beta=2)
        # Question answer layer to output spans of question answers
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        verbose=False
    ):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #B = Batch Size, S = Sequence Length, H = Hidden Size
        #outputs_encoder = (last_hidden_state: (BxSxH), pooler_output:(BxH), hidden_states: (BxSxH))
        outputs_encoder = self.bert_enc(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )
        last_hidden_state, pooler_output, enc_hidden_states = outputs_encoder

        outputs_VQVAE = self.vqvae_model(last_hidden_state)

        # #VQVAE returns: embedding_loss, x_hat, perplexity, z_q, e_indices
        vq_embedding_loss, hidden_states_recon, vqvae_ppl, _, min_encoding_indices = outputs_VQVAE

        # outputs_decoder = self.bert_dec(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         token_type_ids=token_type_ids,
        #         position_ids=position_ids,
        #         head_mask=head_mask,
        #         inputs_embeds=inputs_embeds,
        #         encoder_hidden_states = last_hidden_state
        #     )

        # sequence_output, dec_pooler_output = outputs_decoder

        # Compute logits
        logits = self.qa_outputs(last_hidden_state)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)


        outputs = (start_logits, end_logits, min_encoding_indices) #+ outputs_encoder_vqvae[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are     outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            # # Compute VQVAE loss
            vq_recon_loss = torch.mean((hidden_states_recon - last_hidden_state)**2) # VQVAE divides this by variance of total training data
            vqvae_loss = vq_recon_loss + vq_embedding_loss

            outputs = (total_loss,vqvae_loss) + outputs
            if verbose:
                log.info('VQVAE emb_loss: {}\tppl: {}'.format(vq_embedding_loss, vqvae_ppl))
                log.info('Recon loss: {}'.format(vq_recon_loss))
                log.info('VQVAE Loss: {}'.format(vqvae_loss))
                log.info('BERT Loss: {}'.format(total_loss))

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
