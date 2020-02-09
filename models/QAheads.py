__all__ = [
           'LinearQAHead',
           'RecurrentQAHead',
]

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Encoder import *
from models.Highway import Highway


is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
       
class LinearQAHead(nn.Module):
    
    def __init__(
                 self,
                 in_size:int=1024,
                 n_labels_qa:int=2,
                 multitask:bool=False,
    ):
        
        super(LinearQAHead, self).__init__()
        self.n_labels = n_labels_qa
        self.qa_outputs = nn.Linear(in_size, self.n_labels)
        self.multitask = multitask
        
        #if multi-task setting
        if self.multitask:
            # subjectivity output layer
            self.sbj_outputs = nn.Linear(in_size, 2)

    def forward(
                self,
                bert_outputs:torch.Tensor, 
                start_positions=None,
                end_positions=None,
    ):
        
        sequence_output = bert_outputs[0]      
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
    
    def __init__(
                 self,
                 max_seq_length:int=512,
                 in_size:int=1024,
                 n_labels_qa:int=2,
                 highway_block:bool=False,
                 multitask:bool=False,
    ):
        
        super(RecurrentQAHead, self).__init__()
        self.n_labels = n_labels_qa
        self.lstm_encoder = EncoderLSTM(max_seq_length)
        
        # if highway connection
        if highway_block:
            self.highway = Highway(in_size)
            
        self.qa_outputs = nn.Linear(in_size, self.n_labels)
        self.multitask = multitask
        
        #if multi-task setting
        if self.multitask:    
            # subjectivity output layer
            self.sbj_outputs = nn.Linear(in_size, 2)

    def forward(
                self,
                bert_outputs:torch.Tensor,
                seq_lengths:torch.Tensor,
                start_positions=None,
                end_positions=None,
    ):
        
        sequence_output = bert_outputs[0]
        hidden_lstm = self.lstm_encoder.init_hidden(sequence_output.shape[0])
        
        # pass bert hidden representations through Bi-LSTM to compute temporal dependencies (and global interactions)
        sequence_output, _ = self.lstm_encoder(sequence_output, seq_lengths, hidden_lstm)
        
        if hasattr(self, 'highway'):
            # pass output of Bi-LSTM through highway-connection (for better information flow)
            sequence_output = self.highway(sequence_output)
        
        # compute classification of answer span
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