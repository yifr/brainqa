from transformers import BertModel, BertTokenizer, BertPreTrainedModel, BertConfig
from models.vqvae import VQVAE

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np

#Do we want an encoder, decoder arch?

class BrainQA(BertPreTrainedModel):
    def __init__(self, config):
        super(BrainQA, self).__init__(config)
        self.num_labels = config.num_labels

        self.config_enc = config.to_dict()
        self.config_enc['output_hidden_states'] = True
        self.config_enc = BertConfig.from_dict(self.config_enc)

        self.bert = BertModel(self.config_enc)
        self.config_dec = config.to_dict()
        self.config_dec['is_decoder'] = True
        self.config_dec = BertConfig.from_dict(self.config_dec)
        
        self.bert_dec = BertModel(self.config_dec)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        
        #number of clusters = 4th
        self.vqvae_model= VQVAE(config.hidden_size, 32, 2, 64, 512, .25)

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
    ):
        #encoder bert
        #outputs_encoder = (last_hidden_state, pooler_output, hidden_states, attentions)
        outputs_encoder = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )

        #give to VQVAE
        #outputs embedding loss, x_hat (encoded embedding?), ppl
        #concat x_hat with hidden states
        outputs_VQVAE = self.vqvae_model(outputs_encoder[0])

        vqvae_hidden_states = torch.cat((outputs_encoder[2][0], outputs_VQVAE[1]), dim=1) # TODO

        #decoder bert
        outputs_decoder = self.bert_dec(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states = vqvae_hidden_states
            )
        # hidden states = outputs[2]
        # attention = outputs[3]
        sequence_output = outputs_decoder[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs_decoder[2:]
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
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)