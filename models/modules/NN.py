__all__ = ['FFNN']

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FFNN(nn.Module):

	  def __init__(self, in_size:int, dropout:float=0.25):
    	super(FFNN, self).__init__()
    	self.in_size = in_size
    	self.dropout = dropout

    	self.fc_1 = nn.Linear(self.in_size, self.in_size)
    	nn.init.xavier_uniform_(fc_1.weight)
		self.fc_2 = nn.Linear(self.in_size, 1)
		nn.init.xavier_uniform_(fc_2.weight)

		self.dropout = nn.Dropout(p=self.dropout)

	def forward(self, X):
		out = self.dropout(self.fc_1(X))
		logits = self.fc_2(out)
		return logits