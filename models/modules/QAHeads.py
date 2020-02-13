__all__ = [
           'LinearQAHead',
           'RecurrentQAHead',
]

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.Encoder import *
from models.modules.Highway import Highway

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
"""
       
class LinearQAHead(nn.Module):
    
    def __init__(
                 self,
                 in_size:int=1024,
                 n_labels_qa:int=2,
                 highway_block:bool=False,
                 multitask:bool=False,
                 n_aux_tasks=None,
    ):
        
        super(LinearQAHead, self).__init__()
        self.n_labels = n_labels_qa
        self.qa_outputs = nn.Linear(in_size, self.n_labels)
        self.multitask = multitask
        self.n_aux_tasks = n_aux_tasks
        
        if highway_block:
            self.highway = Highway(in_size)
        
        if self.multitask:
            # subjectivity output layer (must be present in every MTL setting)
            self.sbj_outputs = nn.Linear(in_size, 1)

            if self.n_aux_tasks == 2:
                # 6 domains in SubjQA plus Wikipedia from SQuAD --> 7 classes
                self.domain_outputs = nn.Linear(in_size, 7)

            elif self.n_aux_tasks == 3:
                # 6 domains in SubjQA plus Wikipedia from SQuAD --> 7 classes
                self.domain_outputs = nn.Linear(in_size, 7)
                # TODO: figure out, whether this task is useful at all
                # SubjQA vs. SQuAD (binary classification whether question-context sequence belongs to SQuAD or SubjQA)
                self.ds_outputs = nn.Linear(in_size, 1)

            elif self.n_aux_tasks > 3:
                raise ValueError("Model cannot perform more than 3 auxiliary tasks.")
                    
    def forward(
                self,
                bert_outputs:torch.Tensor, 
                start_positions=None,
                end_positions=None,
    ):
        
        sequence_output = bert_outputs[0]
        
        if hasattr(self, 'highway'):
            # pass BERT representations through highway-connection (for better information flow)
            sequence_output = self.highway(sequence_output)
        
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
            
        if self.multitask and isinstance(self.n_aux_tasks, int):
            sbj_logits = self.sbj_outputs(sequence_output)
            # transform shape of logits from [batch_size, 1] to [batch_size] (necessary for passing logits to loss function)
            sbj_logits = sbj_logits.squeeze(-1)

            if self.n_aux_tasks == 1:
                return outputs, sbj_logits

            elif self.n_aux_tasks == 2:
                domain_logits = self.domain_outputs(sequence_output)
                domain_logits = domain_logits.squeeze(-1)
                return outputs, sbj_logits, domain_logits

            elif self.n_aux_tasks == 3:
                domain_logits = self.domain_outputs(sequence_output)
                ds_logits = self.ds_outputs(sequence_output)
                domain_logits = domain_logits.squeeze(-1)
                ds_logits = ds_logits.squeeze(-1)
                return outputs, sbj_logits, domain_logits, ds_logits
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
                 n_aux_tasks=None,
    ):
        
        super(RecurrentQAHead, self).__init__()
        self.n_labels = n_labels_qa
        self.lstm_encoder = EncoderLSTM(max_seq_length)
        
        if highway_block:
            self.highway = Highway(in_size)
            
        self.qa_outputs = nn.Linear(in_size, self.n_labels)
        self.multitask = multitask
        self.n_aux_tasks = n_aux_tasks
        
        if self.multitask:
            # subjectivity output layer (must be present in every MTL setting)
            self.sbj_outputs = nn.Linear(in_size, 1)

            if self.n_aux_tasks == 2:
                # 6 domains in SubjQA plus Wikipedia from SQuAD --> 7 classes
                self.domain_outputs = nn.Linear(in_size, 7)

            elif self.n_aux_tasks == 3:
                # 6 domains in SubjQA plus Wikipedia from SQuAD --> 7 classes
                self.domain_outputs = nn.Linear(in_size, 7)
                # TODO: figure out, whether this task is useful at all
                # SubjQA vs. SQuAD (binary classification whether question-context sequence belongs to SQuAD or SubjQA)
                self.ds_outputs = nn.Linear(in_size, 1)

            elif self.n_aux_tasks > 3:
                raise ValueError("Model cannot perform more than 3 auxiliary tasks.")

    def forward(
                self,
                bert_outputs:torch.Tensor,
                seq_lengths:torch.Tensor,
                start_positions=None,
                end_positions=None,
    ):
        
        sequence_output = bert_outputs[0]
        hidden_lstm = self.lstm_encoder.init_hidden(sequence_output.shape[0])
        
        # pass BERT representations through Bi-LSTM to compute temporal dependencies and global interactions
        sequence_output, _ = self.lstm_encoder(sequence_output, seq_lengths, hidden_lstm)
        
        if hasattr(self, 'highway'):
            # pass output of Bi-LSTM through highway-connection (for better information flow)
            # TODO: figure out, whether we should pass sequence_output[:, -1, :] to Highway layer or sequence_output
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
        
        if self.multitask and isinstance(self.n_aux_tasks, int):
            # we only need hidden states of last time step (summary of the sequence) (i.e., seq[batch_size, -1, hidden_size])
            sbj_logits = self.sbj_outputs(sequence_output[:, -1, :])
            # transform shape of logits from [batch_size, 1] to [batch_size] (necessary for passing logits to loss function)
            sbj_logits = sbj_logits.squeeze(-1)

            if self.n_aux_tasks == 1:
                return outputs, sbj_logits

            elif self.n_aux_tasks == 2:
                domain_logits = self.domain_outputs(sequence_output[:, -1, :])
                domain_logits = domain_logits.squeeze(-1)
                return outputs, sbj_logits, domain_logits

            elif self.n_aux_tasks == 3:
                domain_logits = self.domain_outputs(sequence_output[:, -1, :])
                ds_logits = self.ds_outputs(sequence_output[:, -1, :])
                domain_logits = domain_logits.squeeze(-1)
                ds_logits = ds_logits.squeeze(-1)
                return outputs, sbj_logits, domain_logits, ds_logits

            else:
                raise ValueError("Model cannot perform more than 3 auxiliary tasks along the main task.")
        else:
            return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)