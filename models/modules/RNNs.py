__all__ = [
           'BiLSTM',
           'BiGRU',
]

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import to_cpu

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
class BiLSTM(nn.Module):
    
    def __init__(
                 self,
                 max_seq_length:int,
                 in_size:int,
                 n_layers:int=2,
                 dropout:float=0.25,
                 bidir:bool=True,
    ):
        
        super(BiLSTM, self).__init__()
        self.in_size = in_size
        self.hidden_size = in_size // 2  # if hidden_size = in_size // 2 -> in_size for classification head is in_size due to bidir
        self.n_layers = n_layers
        self.dropout = dropout
        self.max_seq_length = max_seq_length
        self.bidir = bidir
        
        self.lstm = nn.LSTM(
                            self.in_size,
                            self.hidden_size,
                            self.n_layers,
                            batch_first=True,
                            dropout=self.dropout,
                            bidirectional=self.bidir,
        )
        
    def forward(
                self,
                bert_outputs:torch.Tensor,
                seq_lengths:torch.Tensor,
                hidden:torch.Tensor,
    ):
        # out, hidden = self.lstm(bert_outputs, hidden)

        # NOTE: we don't want to include [PAD] tokens in the recurrent step
        #       (not useful to add hidden_i, where i is the last position of a non-[PAD] token, and input of 0s together) 
        seq_lengths = to_cpu(seq_lengths, detach=True)
        packed = nn.utils.rnn.pack_padded_sequence(bert_outputs, seq_lengths, batch_first=True)
        packed_out, hidden = self.lstm(packed, hidden)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=self.max_seq_length)
        return out, hidden
    
    def init_hidden(
                    self,
                    batch_size:int,
    ):
        # NOTE: we need to initialise twice as many hidden states for bidirectional RNNs
        n = self.n_layers * 2 if self.bidir else self.n_layers 
        hidden_state = torch.zeros(n, batch_size, self.hidden_size, device=device)
        cell_state = torch.zeros(n, batch_size, self.hidden_size, device=device)
        hidden = (nn.init.xavier_uniform_(hidden_state), nn.init.xavier_uniform_(cell_state))
        return hidden
    
class BiGRU(nn.Module):
    
    def __init__( 
                 self,
                 max_seq_length:int,
                 in_size:int,
                 n_layers:int=2,
                 dropout:float=0.25,
                 bidir:bool=True,
    ):
        
        super(BiGRU, self).__init__()
        self.in_size = in_size
        self.hidden_size = in_size // 2 # if hidden_size = in_size // 2 -> in_size for classification head is in_size due to bidir
        self.n_layers = n_layers
        self.dropout = dropout
        self.max_seq_length = max_seq_length
        self.bidir = bidir
        
        self.gru = nn.GRU(
                          self.in_size,
                          self.hidden_size,
                          self.n_layers,
                          batch_first=True,
                          dropout=self.dropout,
                          bidirectional=self.bidir,
        )
        
                
    def forward(
                self,
                bert_outputs:torch.Tensor,
                seq_lengths:torch.Tensor,
                hidden:torch.Tensor,
    ):
        # out, hidden = self.gru(bert_outputs, hidden)
        
        # NOTE: we don't want to include [PAD] tokens in the recurrent step
        #       (not useful to add hidden_i, where i is the last position of a non-[PAD] token, and input of 0s together) 
        seq_lengths = to_cpu(seq_lengths, detach=True)        
        packed = nn.utils.rnn.pack_padded_sequence(bert_outputs, seq_lengths, batch_first=True)
        packed_out, hidden = self.gru(packed, hidden)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=self.max_seq_length)
        return out, hidden
    
    def init_hidden(
                    self,
                    batch_size:int,
    ):
        # NOTE: we need to initialise twice as many hidden states for bidirectional RNNs
        n = self.n_layers * 2 if self.bidir else self.n_layers 
        # NOTE: in contrast to LSTMs, GRUs don't need cell state initialisations (GRUs work similar to simple Elman RNNs)
        hidden_state = torch.zeros(n, batch_size, self.hidden_size, device=device)
        return nn.init.xavier_uniform_(hidden_state)