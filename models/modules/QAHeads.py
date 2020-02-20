__all__ = [
           'LinearQAHead',
           'RecurrentQAHead',
]

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.RNNs import *
from models.modules.GradReverse import *
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
                 in_size:int,
                 n_labels_qa:int,
                 qa_dropout_p:float,
                 highway_block:bool=False,
                 multitask:bool=False,
                 n_aux_tasks=None,
                 aux_dropout_p:float=0.25,
                 n_domain_labels=None,
                 adversarial:bool=False,
    ):
        
        super(LinearQAHead, self).__init__()
        self.in_size = in_size
        self.n_labels = n_labels_qa
        assert self.n_labels == 2
        self.multitask = multitask
        self.n_aux_tasks = n_aux_tasks
        self.aux_dropout_p = aux_dropout_p
        self.qa_dropout_p = qa_dropout_p 

        if highway_block:
            self.highway = Highway(self.in_size)

        # fully-connected QA output layer with dropout
        self.fc_qa = nn.Linear(self.in_size, self.n_labels)
        self.qa_dropout = nn.Dropout(p = self.qa_dropout_p)
        
        if self.multitask:

            # define, whether we want to perform adversarial training with a GRL between feature extractor and classifiers
            self.adversarial = adversarial

            # define dropout layer for auxiliary classification tasks
            self.aux_dropout = nn.Dropout(p = self.aux_dropout_p)
            
            # fully-connected subjectivity output layers (must be present in every MTL setting)
            self.fc_sbj = nn.Linear(self.in_size, self.in_size // 2)
            self.fc_sbj_a = nn.Linear(self.in_size // 2, 1) # fc subj. layer for answers
            self.fc_sbj_q = nn.Linear(self.in_size // 2, 1) # fc subj. layer for questions

            if self.n_aux_tasks == 2:
                assert isinstance(n_domain_labels, int), 'If model is to perform two auxiliary tasks, domain labels must be provided'
                self.n_domain_labels = n_domain_labels

                # fully-connected review domain output layers (second auxiliary task)
                self.fc_domain_1 = nn.Linear(self.in_size, self.in_size // 2)
                self.fc_domain_2 = nn.Linear(self.in_size // 2, self.n_domain_labels)

            elif self.n_aux_tasks == 3:
                assert isinstance(n_domain_labels, int), 'If model is to perform three auxiliary tasks, domain labels must be provided'
                self.n_domain_labels = n_domain_labels

                self.fc_domain_1 = nn.Linear(self.in_size, self.in_size // 2)
                self.fc_domain_2 = nn.Linear(self.in_size // 2, self.n_domain_labels)

                # TODO: figure out, whether third auxiliary task is useful at all
                # SubjQA vs. SQuAD (binary classification whether question-context sequence belongs to SQuAD or SubjQA)
                self.fc_ds_1 = nn.Linear(self.in_size, self.in_size // 2)
                self.fc_ds_2 = nn.Linear(self.in_size // 2, 1)

            elif self.n_aux_tasks > 3:
                raise ValueError("Model cannot perform more than 3 auxiliary tasks.")
                    
    def forward(
                self,
                distilbert_output:torch.Tensor,
                task:str, 
                start_positions=None,
                end_positions=None,
    ):
        sequence_output = distilbert_output[0]
        sequence_output = self.qa_dropout(sequence_output)
        
        if hasattr(self, 'highway'):
            sequence_output = self.highway(sequence_output) # pass BERT representations through highway-connection (for better information flow)
       
        if task == 'QA':

            logits = self.fc_qa(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            outputs = (start_logits, end_logits,) + distilbert_output[1:]

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

                loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2
                outputs = (total_loss,) + outputs

            return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)

        else:

            # use contextual embedding of the special [CLS] token (corresponds to the semantic representation of an input sentence X)
            sequence_output = sequence_output[:, 0, :] 

            if self.adversarial:
                # reverse gradients to learn qa-type / domain-invariant features (i.e., semi-supervised domain-adaptation)
                sequence_output = grad_reverse(sequence_output)

            if task == 'Sbj_Class':

                sbj_out = F.relu(self.aux_dropout(self.fc_sbj(sequence_output)))
                sbj_logits_a = self.fc_sbj_a(sbj_out)
                sbj_logits_q = self.fc_sbj_q(sbj_out)

                # transform shape of logits from [batch_size, 1] to [batch_size] (necessary for passing logits to loss function)
                sbj_logits_a = sbj_logits_a.squeeze(-1)
                sbj_logits_q = sbj_logits_q.squeeze(-1)

                return sbj_logits_a, sbj_logits_q

            elif task == 'Domain_Class':

                domain_out = F.relu(self.aux_dropout(self.fc_domain_1(sequence_output)))
                domain_logits = self.fc_domain_2(domain_out)
                domain_logits = domain_logits.squeeze(-1)

                return domain_logits
        
        
class RecurrentQAHead(nn.Module):
    
    def __init__(
                 self,
                 in_size:int,
                 n_labels_qa:int,
                 qa_dropout_p:float,
                 max_seq_length:int=512,
                 highway_block:bool=False,
                 decoder:bool=False,
                 multitask:bool=False,
                 n_aux_tasks=None,
                 aux_dropout_p:float=0.25,
                 n_domain_labels=None,
                 adversarial:bool=False,
    ):
        super(RecurrentQAHead, self).__init__()
        
        self.in_size = in_size
        self.n_labels = n_labels_qa
        assert self.n_labels == 2
        self.qa_dropout_p = qa_dropout_p
        self.multitask = multitask
        self.n_aux_tasks = n_aux_tasks
        self.aux_dropout_p = aux_dropout_p
        self.n_recurrent_layers = 2 # set number of recurrent layers to 1 or 2 (more are not necessary and computationally inefficient / costly)
        self.rnn_version = 'LSTM'

        self.rnn_encoder = BiLSTM(max_seq_length, in_size=self.in_size, n_layers=self.n_recurrent_layers) if self.rnn_version == 'LSTM' else BiGRU(max_seq_length, in_size=self.in_size, n_layers=self.n_recurrent_layers)
        
        if highway_block:
            self.highway = Highway(self.in_size) # highway bridge in-between bidirectional RNNs

        if decoder:
            self.rnn_decoder = BiLSTM(max_seq_length, in_size=self.in_size, n_layers=self.n_recurrent_layers) if self.rnn_version == 'LSTM' else BiGRU(max_seq_length, in_size=self.in_size, n_layers=self.n_recurrent_layers)
            
        # fully-connected QA output layer with dropout
        self.fc_qa = nn.Linear(self.in_size, self.n_labels)
        self.qa_dropout = nn.Dropout(self.qa_dropout_p)
        
        if self.multitask:
            # define, whether we want to perform adversarial training with a GRL between feature extractor and classifiers
            self.adversarial = adversarial

            # define dropout layer for auxiliary classification tasks
            self.aux_dropout = nn.Dropout(p = self.aux_dropout_p)
            
            # fully-connected subjectivity output layers (must be present in every MTL setting)
            self.fc_sbj = nn.Linear(self.in_size, self.in_size // 2)
            self.fc_sbj_a = nn.Linear(self.in_size // 2, 1) # fc subj. layer for answers
            self.fc_sbj_q = nn.Linear(self.in_size // 2, 1) # fc subj. layer for questions

            if self.n_aux_tasks == 2:
                assert isinstance(n_domain_labels, int), 'If model is to perform two auxiliary tasks, domain labels must be provided'
                self.n_domain_labels = n_domain_labels

                # fully-connected review domain output layers (second auxiliary task)
                self.fc_domain_1 = nn.Linear(self.in_size, self.in_size // 2)
                self.fc_domain_2 = nn.Linear(self.in_size // 2, self.n_domain_labels)

            elif self.n_aux_tasks == 3:
                assert isinstance(n_domain_labels, int), 'If model is to perform three auxiliary tasks, domain labels must be provided'
                self.n_domain_labels = n_domain_labels

                self.fc_domain_1 = nn.Linear(self.in_size, self.in_size // 2)
                self.fc_domain_2 = nn.Linear(self.in_size // 2, self.n_domain_labels)

                # TODO: figure out, whether third auxiliary task is at all usefull
                # SubjQA vs. SQuAD (binary classification whether question-review sequence belongs to SQuAD or SubjQA)
                self.fc_ds_1 = nn.Linear(self.in_size, self.in_size // 2)
                self.fc_ds_2 = nn.Linear(self.in_size // 2, 1)

            elif self.n_aux_tasks > 3:
                raise ValueError("Model cannot perform more than 3 auxiliary tasks.")

    def forward(
                self,
                distilbert_output:torch.Tensor,
                seq_lengths:torch.Tensor,
                task:str,
                start_positions=None,
                end_positions=None,
    ):
        sequence_output = distilbert_output[0]
        sequence_output = self.qa_dropout(sequence_output)

        hidden_rnn = self.rnn_encoder.init_hidden(sequence_output.shape[0])
        
        # pass BERT representations through Bi-LSTM or Bi-GRU to compute both temporal dependencies and global interactions among feature representations
        sequence_output, hidden_rnn = self.rnn_encoder(sequence_output, seq_lengths, hidden_rnn)
        
        if hasattr(self, 'highway'):
            # pass output of Bi-LSTM or Bi-GRU through a Highway connection (for better information flow)
            # TODO: figure out, whether we should pass "sequence_output[:, -1, :]" to Highway layer or simply "sequence_output"
            sequence_output = self.highway(sequence_output)

        if hasattr(self, 'rnn_decoder'):
            sequence_output, hidden_rnn = self.rnn_decoder(sequence_output, seq_lengths, hidden_rnn)
        

        if task == 'QA':

            logits = self.fc_qa(sequence_output)

            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            outputs = (start_logits, end_logits,) + distilbert_output[1:]

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

                loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2
                outputs = (total_loss,) + outputs

            return outputs  #, hidden_rnn  # (loss), start_logits, end_logits, (hidden_states), (attentions)

        else:
            if self.adversarial:
                # reverse gradients to learn qa-type / domain-invariant features (i.e., semi-supervised domain-adaptation)
                sequence_output = grad_reverse(sequence_output)

            if task == 'Sbj_Class':

                # we only need hidden states of last time step (summary of the sequence) (i.e., seq[batch_size, -1, hidden_size])
                sbj_out = F.relu(self.aux_dropout(self.fc_sbj(sequence_output[:, -1, :])))
                sbj_logits_a = self.fc_sbj_a(sbj_out)
                sbj_logits_q = self.fc_sbj_q(sbj_out)

                # transform shape of logits from [batch_size, 1] to [batch_size] (necessary for passing logits to loss function)
                sbj_logits_a = sbj_logits_a.squeeze(-1)
                sbj_logits_q = sbj_logits_q.squeeze(-1)

                return sbj_logits_a, sbj_logits_q #, hidden_rnn

            elif task == 'Domain_Class':

                domain_out = F.relu(self.aux_dropout(self.fc_domain_1(sequence_output[:, -1, :])))
                domain_logits = self.fc_domain_2(domain_out)
                domain_logits = domain_logits.squeeze(-1)

                return domain_logits #, hidden_rnn