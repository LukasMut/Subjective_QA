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
    parser.add_argument('--finetuning',  type=str, default='SQuAD',
            help='If SQuAD, fine tune on SQuAD only; if SubjQA, fine tune on SubjQA only; if both, fine tune on both SQuAD and SubjQA.')
    parser.add_argument('--version',  type=str, default='train',
            help='If train, then train model on train set(s); if test, then evaluate model on test set(s).')
    parser.add_argument('--multitask', action='store_true',
            help='If provided, MTL instead of STL setting.')
    parser.add_argument('--n_tasks', type=int, default=1,
            help='Define number of tasks QA model should be trained on. Only necessary, if MTL setting.')
    parser.add_argument('--qa_head', type=str, default='linear',
            help='If linear, put fc linear head on top of BERT; if recurrent, put BiLSTM encoder plus fc linear head on top of BERT.')
    parser.add_argument('--highway_connection', action='store_true',
            help='If provided, put highway connection in between BiLSTM encoder and fc linear output head; NOT relevant for linear head')
    parser.add_argument('--bert_weights', type=str, default='cased',
            help='If cased, load pretrained weights from BERT cased model; if uncased, load pretrained weights from BERT uncased model.')
    parser.add_argument('--batch_size', type=int, default=32,
            help='Define mini-batch size.')
    parser.add_argument('--sd', type=str, default='',
            help='set model save directory for QA model.')
    args = parser.parse_args()
    print(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    if (args.finetuning == 'SubjQA') or (args.finetuning == 'both'):
        
        if args.version == 'train':
        
            subjqa_data_train = get_data(
                                         source='/SubjQA/',
                                         split='/train',
                                         domain='all',
            )

            subjqa_data_dev = get_data(
                                       source='/SubjQA/',
                                       split='/dev',
                                       domain='all',
            )
            
            # convert pd.DataFrames into list of dictionaries (as many dicts as examples)
            subjqa_data_train = convert_df_to_dict(
                                                   subjqa_data_train,
                                                   split='train',
            )
            subjqa_data_dev = convert_df_to_dict(
                                                 subjqa_data_dev,
                                                 split='dev',
            )
            
            # convert dictionaries into instances of preprocessed question-answer-review examples    
            subjqa_examples_train = create_examples(
                                                subjqa_data_train,
                                                source='SubjQA',
                                                is_training=True,
            )

            subjqa_examples_dev = create_examples(
                                                  subjqa_data_dev,
                                                  source='SubjQA',
                                                  is_training=True,
            )

        elif args.version == 'test':
            
            subjqa_data_test = get_data(
                                        source='/SubjQA/',
                                        split='/test',
                                        domain='all',
            )
            
            subjqa_data_test = convert_df_to_dict(
                                                  subjqa_data_test,
                                                  split='test',
            )
            
            subjqa_examples_test = create_examples(
                                                   subjqa_data_test,
                                                   source='SubjQA',
                                                   is_training=True,
            )
            
        else:
            raise ValueError('Version of experiment must be one of {train, test}')
        
    elif (args.finetuning == 'SQuAD') or (args.finetuning == 'both'):
        
        if args.version == 'train':
            
            squad_data_train = get_data(
                                        source='/SQuAD/',
                                        split='train',
            )
            
            squad_examples_train = create_examples(
                                       squad_data_train,
                                       source='SQuAD',
                                       is_training=True,
            )

            # create train and dev examples from SQuAD train set only
            squad_examples_train, squad_examples_dev = split_into_train_and_dev(squad_examples_train)
   
    # TODO: figure out, whether we should use pretrained weights from 'bert-base-cased' or 'bert-base-uncased' model
    if args.bert_weights == 'cased':
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        pretrained_weights = 'bert-large-cased-whole-word-masking-finetuned-squad'
    elif args.bert_weights == 'uncased':
        bert_tokenizer == BertTokenizer.from_pretrained('bert-base-uncased')
        pretrained_weights = 'bert-large-uncased-whole-word-masking-finetuned-squad'
    else:
        raise ValueError('Pretrained weights must be loaded from an uncased or cased BERT model')
    
    # set hyperparameters
    max_seq_length = 512 # BERT cannot deal with sequences, where T > 512
    batch_size = args.batch_size
    
    # create domain_to_idx and dataset_to_idx mappings
    domains = ['books', 'electronics', 'grocery', 'movies', 'restaurants', 'tripadvisor', 'all', 'wikipedia']
    datasets = ['SQuAD', 'SubjQA']
    idx_to_domain = dict(enumerate(domains))
    domain_to_idx = {domain: idx for idx, domain in enumerate(domains)}
    idx_to_dataset = dict(enumerate(datasets))
    dataset_to_idx = {dataset: idx for idx, dataset in enumerate(datasets)}
    
    