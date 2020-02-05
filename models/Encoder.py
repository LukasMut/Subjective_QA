__all__ = [
           'EncoderLSTM',
           'EncoderGRU',
]

import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.cuda.is_available() checks and returns True if a GPU is available, else it'll return False
#is_cuda = torch.cuda.is_available()

#if is_cuda:
#    device = torch.device("cuda")
#    print("GPU is available")
#else:

device = torch.device("cpu")
print("GPU not available, CPU used")

class EncoderLSTM(nn.Module):
    
    def __init__(
                 self,
                 max_seq_length:int,
                 in_size:int=1024,
                 n_layers:int=2,
                 dropout:float=0.3,
                 bidir:bool=True,
    ):
        
        super(EncoderLSTM, self).__init__()
        self.in_size = in_size # dimensionality of BERT-large attention heads (i.e, 1024)
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
        #TODO: figure out, whether rnn.pack_padded_sequence is useful for QA (most likely, since we have padded seqs in each batch)
        out, hidden = self.lstm(bert_outputs, hidden)
        
        #seq_lengths = seq_lengths.detach().cpu().numpy()
        
        #NOTE: set enforce_sorted to True, if you need ONNX exportability (sequences must be passed in decreasing order wrt length)
        #packed = nn.utils.rnn.pack_padded_sequence(bert_outputs, seq_lengths, batch_first=True, enforce_sorted=False)
        #out, hidden = self.lstm(packed, hidden)
        #out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=self.max_seq_length)
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
    
class EncoderGRU(nn.Module):
    
    def __init__( 
                 self,
                 max_seq_length:int,
                 in_size:int=1024,
                 n_layers:int=2,
                 dropout:float=0.3,
                 bidir:bool=True,
    ):
        
        super(EncoderGRU, self).__init__()
        self.in_size = in_size # dimensionality of BERT-large attention heads (i.e, 1024)
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
        
        #TODO: figure out, whether rnn.pack_padded_sequence is useful for QA (most likely, since we have padded seqs in each batch)
        out, hidden = self.gru(bert_outputs, hidden)
        
        #seq_lengths = seq_lengths.detach().cpu().numpy()
        
        #NOTE: set enforce_sorted to True, if you need ONNX exportability (sequences must be passed in decreasing order wrt length)
        
        #packed = nn.utils.rnn.pack_padded_sequence(bert_outputs, seq_lengths, batch_first=True)
        #out, hidden = self.gru(packed, hidden)
        #out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=self.max_seq_length)
        return out, hidden
    
    def init_hidden(
                    self,
                    batch_size:int,
    ):
        # NOTE: we need to initialise twice as many hidden states for bidirectional RNNs
        n = self.n_layers * 2 if self.bidir else self.n_layers 
        # NOTE: opposed to LSTM, GRUs don't need cell state (GRUs work similar to simple Elman RNNs)
        hidden_state = torch.zeros(n, batch_size, self.hidden_size, device=device)
        return nn.init.xavier_uniform_(hidden_state)