import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

import argparse
import datetime
import json
import os
import re
import torch 
import transformers

from collections import Counter, defaultdict
from tqdm import trange, tqdm
from transformers import BertTokenizer, BertModel, BertForQuestionAnswering

from eval_squad import *
from models.QAModels import *
from models.utils import *
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',  type=str, default='SQuAD',
            help='If SQuAD, fine tune on SQuAD only; if SubjQA, fine tune on SubjQA only; if both, fine tune on both SQuAD and SubjQA.')
    parser.add_argument('--multitask', action='store_true',
            help='If provided, MTL instead of STL setting.')
    parser.add_argument('--sd', type=str, default='',
            help='set model save directory for QA model.')
    parser.add_argument('--qa_head', type=str, default='linear',
            help='If linear, fc linear head on top of BERT; if recurrent, Bi-LSTM encoder plus fc linear head on top of BERT.')
    parser.add_argument('--BERT', action='store_true',
            help='If provided: Use BERT context embeddings, Else: use GloVe embeddings')
    parser.add_argument('--merged', type=str, default='avg',
            help='max: use max-pooled bert embeddings, avg: use averaged bert embeddings, sum: use summed bert embeddings')
    parser.add_argument('--POS', action='store_true',
            help='Add Part-of-Speech embeddings')
    args = parser.parse_args()
    print(args)