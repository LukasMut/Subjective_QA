__all__ = ['FFNN']

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FFNN(nn.Module):

	def __init__(self, in_size:int):
		super(FFNN, self).__init__()
		self.in_size = in_size
		self.fc_1 = nn.Linear(self.in_size, self.in_size)
		self.fc_2 = nn.Linear(self.in_size, 1)
		nn.init.xavier_uniform_(self.fc_1.weight)
		nn.init.xavier_uniform_(self.fc_2.weight)

	def forward(self, x):
		logits = self.fc_2(F.relu(self.fc_1(x)))
		logits = logits.squeeze(-1)
		return logits