__all__ = ['FFNN']

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FFNN(nn.Module):

	  def __init__(self, in_size:int):
    	super(FFNN, self).__init__()
    	self.insize = insize
		self.fc = nn.Linear(self.in_size, 1)
		nn.init.xavier_uniform_(fc.weight)

	def forward(self, X):
		logits = self.fc(X)
		return logits