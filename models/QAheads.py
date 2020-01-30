__all__ = ['LinearQAHead']

import torch
import torch.nn.functional as F

from torch import nn
from models.Encoder import *
       
class LinearQAHead(nn.Module):
    
    def __init__(self, in_size:int=1024, n_labels_qa:int=2, multitask:bool=False):
        super(LinearQAHead, self).__init__()
        self.n_labels = n_labels_qa
        self.qa_outputs = nn.Linear(in_size, n_labels_qa)
        self.multitask = False
        
        #if multi-task setting
        if multitask:
            self.multitask = True
            self.sbj_outputs = nn.Linear(in_size, 5)

    def forward(self, bert_outputs, start_positions=None, end_positions=None):
        sequence_output = bert_outputs[0]
        # NOTE: figure out, whether view operation in line below is necessary (linear layers can also deal with 3D inputs)
        sequence_output = sequence_output.view(-1, sequence_output.shape[-1])
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + bert_outputs[2:]
        if (start_positions is not None) and (end_positions is not None):
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
        
        if self.multitask:
            sbj_logits = self.sbj_outputs(sequence_output)
            return outputs, sbj_logits
        else:
            return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
        
        
class RecurrentQAHead(nn.Module):
    
    def __init__(self, max_source_length:int, in_size:int=1024, n_labels_qa:int=2, multitask:bool=False):
        super(RecurrentQAHead, self).__init__()
        self.n_labels = n_labels_qa
        self.lstm_encoder = EncoderLSTM(max_source_length)
        self.qa_outputs = nn.Linear(in_size, n_labels_qa)
        self.multitask = False
        
        #if multi-task setting
        if multitask:
            self.multitask = True
            self.sbj_outputs = nn.Linear(in_size, 5)

    def forward(self, bert_outputs, qa_lengths, start_positions=None, end_positions=None):
        
        sequence_output = bert_outputs[0]
        hidden_lstm = self.lstm_encoder.init_hidden(sequence_output.shape[0])
        
        # pass bert hidden representations through bi-lstm to compute temporal dependencies
        sequence_output, _ = self.lstm_encoder(sequence_output, qa_lengths, hidden_lstm)
        
        # NOTE: figure out, whether view operation in line below is necessary (linear layers can also deal with 3D inputs)
        sequence_output = sequence_output.view(-1, sequence_output.shape[-1])
        logits = self.qa_outputs(sequence_output)
        
        # split logits into chunks for start and end of answer span respectively
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + bert_outputs[2:]
        if (start_positions is not None) and (end_positions is not None):
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
        
        if self.multitask:
            sbj_logits = self.sbj_outputs(sequence_output)
            return outputs, sbj_logits
        else:
            return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)

