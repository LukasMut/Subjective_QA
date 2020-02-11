from __future__ import absolute_import, division, print_function

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import argparse
import datetime
import json
import os
import random
import re
import torch 
import transformers

from collections import Counter, defaultdict
from torch.optim import Adam, SGD, CosineAnnealingWarmRestarts
from transformers import BertTokenizer, BertModel, BertForQuestionAnswering
from transformers import AdamW
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup, get_linear_schedule_with_warmup

from eval_squad import *
from models.QAModels import *
from models.utils import *
from utils import *

# set random seeds to reproduce results
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--finetuning', type=str, default='SQuAD',
            help='If SQuAD, fine tune on SQuAD only; if SubjQA, fine tune on SubjQA only; if combined, fine tune on both SQuAD and SubjQA simultaneously.')
    parser.add_argument('--version', type=str, default='train',
            help='If train, then train model on train set(s); if test, then evaluate model on SubjQA test set.')
    parser.add_argument('--multitask', action='store_true',
            help='If provided, MTL instead of STL setting.')
    parser.add_argument('--n_tasks', type=int, default=1,
            help='Define number of tasks QA model should be trained on. Only necessary, if MTL setting.')
    parser.add_argument('--qa_head', type=str, default='linear',
            help='If linear, put fc linear head on top of BERT; if recurrent, put BiLSTM encoder plus fc linear head on top of BERT.')
    parser.add_argument('--highway_connection', action='store_true',
            help='If provided, put highway connection in between BERT OR BiLSTM encoder and fc linear output head.')
    parser.add_argument('--bert_weights', type=str, default='cased',
            help='If cased, load pretrained weights from BERT cased model; if uncased, load pretrained weights from BERT uncased model.')
    parser.add_argument('--batch_size', type=int, default=32,
            help='Define mini-batch size.')
    parser.add_argument('--n_epochs', type=int, default=3,
            help='Set number of epochs model should train for. Should be a higher number, if we fine-tune on SubjQA only.')
    parser.add_argument('--optim', type=str, default='AdamW',
            help='Define optimizer. Must be one of {AdamW, Adam, SGD}.')
    parser.add_argument('--sd', type=str, default='',
            help='set model save directory for QA model.')
    parser.add_argument('--not_finetuned', action='store_true',
            help='If provided, test pre-trained BERT large model, fine-tuned on SQuAD, on SubjQA (no prior task-specific fine-tuning)')
    args = parser.parse_args()
    
    # see whether arg.parser works correctly
    print(args)
    print()
    
    # move model and tensors to GPU, if GPU is available (device must be defined)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # set some crucial hyperparameters
    max_seq_length = 512 # BERT cannot deal with sequences, where T > 512
    doc_stride = 200
    max_query_length = 50
    batch_size = args.batch_size
    
    # create domain_to_idx and dataset_to_idx mappings (necessary for auxiliary tasks)
    domains = ['books', 'electronics', 'grocery', 'movies', 'restaurants', 'tripadvisor', 'all', 'wikipedia']
    datasets = ['SQuAD', 'SubjQA']
    idx_to_domain = dict(enumerate(domains))
    domain_to_idx = {domain: idx for idx, domain in enumerate(domains)}
    idx_to_dataset = dict(enumerate(datasets))
    dataset_to_idx = {dataset: idx for idx, dataset in enumerate(datasets)}
    
     # TODO: figure out, whether we should use pretrained weights from 'bert-base-cased' or 'bert-base-uncased' model
    if args.bert_weights == 'cased':
        
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        pretrained_weights = 'bert-large-cased-whole-word-masking-finetuned-squad'
        
    elif args.bert_weights == 'uncased':
        
        bert_tokenizer == BertTokenizer.from_pretrained('bert-base-uncased')
        pretrained_weights = 'bert-large-uncased-whole-word-masking-finetuned-squad'
        
    else:
        raise ValueError('Pretrained weights must be loaded from an uncased or cased BERT model.')
                    
    qa_head_name = 'RecurrentQAHead' if args.qa_head == 'recurrent' else 'LinearQAHead'
    highway = 'Highway' if args.highway_connection else ''
    train_method = 'multitask' + '_' + str(args.n_tasks) if args.multitask else 'singletask'
    model_name = 'BERT' + '_' + args.bert_weights + '_' + qa_head_name + '_' + highway + '_' + train_method
    
    if args.version == 'train':
        
        if args.finetuning == 'SubjQA' or args.finetuning == 'combined':
        
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
            
            subjqa_features_train = convert_examples_to_features(
                                                                 subjqa_examples_train, 
                                                                 bert_tokenizer,
                                                                 max_seq_length=max_seq_length,
                                                                 doc_stride=doc_stride,
                                                                 max_query_length=max_query_length,
                                                                 is_training=True,
                                                                 domain_to_idx=domain_to_idx,
                                                                 dataset_to_idx=dataset_to_idx,
            )

            subjqa_features_dev = convert_examples_to_features(
                                                               subjqa_examples_dev, 
                                                               bert_tokenizer,
                                                               max_seq_length=max_seq_length,
                                                               doc_stride=doc_stride,
                                                               max_query_length=max_query_length,
                                                               is_training=True,
                                                               domain_to_idx=domain_to_idx,
                                                               dataset_to_idx=dataset_to_idx,
            )
            
            subjqa_tensor_dataset_train = create_tensor_dataset(
                                                                subjqa_features_train,
                                                                evaluate=False,
            )

            subjqa_tensor_dataset_dev = create_tensor_dataset(
                                                              subjqa_features_dev,
                                                              evaluate=False,
            )
                
        elif args.finetuning == 'SQuAD' or args.finetuning == 'combined':
            
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
            
            
            squad_features_train = convert_examples_to_features(
                                                                squad_examples_train, 
                                                                bert_tokenizer,
                                                                max_seq_length=max_seq_length,
                                                                doc_stride=doc_stride
                                                                max_query_length=max_query_length,
                                                                is_training=True,
                                                                domain_to_idx=domain_to_idx,
                                                                dataset_to_idx=dataset_to_idx,
            )

            squad_features_dev = convert_examples_to_features(
                                                             squad_examples_dev, 
                                                             bert_tokenizer,
                                                             max_seq_length=max_seq_length,
                                                             doc_stride=doc_stride
                                                             max_query_length=max_query_length,
                                                             is_training=True,
                                                             domain_to_idx=domain_to_idx,
                                                             dataset_to_idx=dataset_to_idx,
            )
            
            squad_tensor_dataset_train = create_tensor_dataset(
                                                   squad_features_train,
                                                   evaluate=False,
            )

            squad_tensor_dataset_dev = create_tensor_dataset(
                                                 squad_features_dev,
                                                 evaluate=False,
            )
        
        if args.finetuning == 'SQuAD':
            
            train_dl = create_batches(
                                      dataset=squad_tensor_dataset_train,
                                      batch_size=batch_size,
                                      split='train',
            )

            val_dl = create_batches(
                                    dataset=squad_tensor_dataset_dev,
                                    batch_size=batch_size,
                                    split='eval',
            )
            
        elif args.finetuning == 'SubjQA':
            
            train_dl = create_batches(
                                      dataset=subjqa_tensor_dataset_train,
                                      batch_size=batch_size,
                                      split='train',
            )

            val_dl = create_batches(
                                    dataset=subjqa_tensor_dataset_dev,
                                    batch_size=batch_size,
                                    split='eval',
            )
                
        elif args.finetuning == 'combined':
            
            train_dl = AlternatingBatchGenerator(
                                                 squad_tensor_dataset_train,
                                                 subjqa_tensor_dataset_train,
                                                 batch_size=batch_size,
                                                 split='train',
            )

            val_dl = AlternatingBatchGenerator(
                                               squad_tensor_dataset_train,
                                               subjqa_tensor_dataset_train,
                                               batch_size=batch_size,
                                               split='eval',
            )
        
        # initialise QA model
        model = BertForQA.from_pretrained(
                                          pretrained_weights,
                                          qa_head_name=qa_head_name,
                                          max_seq_length=max_seq_length,
                                          highway_connection=args.highway_connection,
                                          multitask=args.multitask,
        )

        # set model to device
        model.to(device)

        hypers = {
                  "lr_adam": 1e-3,
                  "lr_sgd": 1e-2,
                  "lr_sgd_cos": 1e-1,
                  "warmup_steps": 50,
                  "max_grad_norm": 10,
                  "sort_batch": False, # TODO: figure out, whether we should sort batch for RNNs (not necessary for linear QA heads)
        }

        hypers["n_epochs"] = args.n_epochs
        hypers["freeze_bert"] = True if args.finetuning == 'SQuAD' or args.finetuning == 'combined' else False
        hypers["optim"] = args.optim
        hypers["model_dir"] = args.sd
        hypers["model_name"] = model_name
        
        if args.optim == 'AdamW':
            
            optimizer = AdamW(
                              model.parameters(), 
                              lr=hypers['lr_adam'], 
                              correct_bias=False,
            )
            
            t_total = len(train_dl) * hypers['n_epochs'] # total number of training steps (i.e., step = iteration)
            
            scheduler = get_linear_schedule_with_warmup(
                                                        optimizer, 
                                                        num_warmup_steps=hypers["warmup_steps"], 
                                                        num_training_steps=t_total,
            )
            
        elif args.optim == 'Adam':
            
            optimizer = Adam(
                             model.parameters(),
                             lr=hypers['lr_adam'],
                             amsgrad=True,
            )
            scheduler = None
        
        elif args.optim == 'SGD':
            
            optimizer = SGD(
                            model.parameters(),
                            lr=hypers['lr_sgd_cos'], 
                            momentum=0.9,
            )
            # TODO: figure out, whether cosine-annealing is useful for SGD with momentum when optimizing a BERT-QA model
            scheduler = None #CosineAnnealingWarmRestarts(
                             #                       optimizer,
                             #                       T_0=10,
                             #                       T_mult=2,
            #)
        
        else:
            raise ValueError("Optimizer must be one of {AdamW, Adam, SGD}.")
                    
        batch_losses, train_losses, train_accs, train_f1s, val_losses, val_accs, val_f1s, model = train(
                                                                                                        model=model,
                                                                                                        tokenizer=bert_tokenizer,
                                                                                                        train_dl=train_dl,
                                                                                                        val_dl=val_dl,
                                                                                                        batch_size=batch_size,
                                                                                                        args=hypers,
                                                                                                        optimizer=optimizer,
                                                                                                        scheduler=scheduler,
                                                                                                        early_stopping=True,
        )
        
        train_results  = dict()
        train_results['batch_losses'] = batch_losses
        train_results['train_losses'] = train_losses
        train_results['train_accs'] = train_accs
        train_results['train_f1s'] = train_f1s
        train_results['val_losses'] = val_losses
        train_results['val_accs'] = val_accs
        train_results['val_f1s'] = val_f1s
            
        with open('./results_train/' + model_name + '.json', 'w') as json_file:
            json.dump(train_results, json_file)
        
        # TODO: implement model unfreezing (necessary for fine-tuning on SubjQA - freeze for ~ 2 epochs, unfreeze, train as long as for other setting)
        
    # we always test on SubjQA
    elif args.version == 'test':
            
            subjqa_data_test = convert_df_to_dict(
                                                  subjqa_data_test,
                                                  split='test',
            )
            
            # convert dictionaries into instances of preprocessed question-answer-review examples    
            subjqa_examples_test = create_examples(
                                                   subjqa_data_test,
                                                   source='SubjQA',
                                                   is_training=True,
            )
            
            subjqa_features_test = convert_examples_to_features(
                                                                subjqa_examples_test, 
                                                                bert_tokenizer,
                                                                max_seq_length=max_seq_length,
                                                                doc_stride=doc_stride,
                                                                max_query_length=max_query_length,
                                                                is_training=True,
                                                                domain_to_idx=domain_to_idx,
                                                                dataset_to_idx=dataset_to_idx,
            )
            
            subjqa_tensor_dataset_test = create_tensor_dataset(
                                                               subjqa_features_test,
                                                               evaluate=False,
            )  
            
            test_dl = create_batches(
                                     dataset=subjqa_tensor_dataset_test,
                                     batch_size=batch_size,
                                     split='eval',
            )
            
            if args.not_finetuned:
                # test (simple) BERT-QA-model fine-tuned on SQuAD without (prior)task-specific fine-tuning on SubjQA
                model = BertForQuestionAnswering.from_pretrained(pretrained_weights)
                model_name = 'BERT_pretrained_SQuAD_no_fine_tuning'
            else:
                model = BertForQA.from_pretrained(
                                                  pretrained_weights,
                                                  qa_head_name=qa_head_name,
                                                  max_seq_length=max_seq_length,
                                                  highway_connection=args.highway_connection,
                                                  multitask=args.multitask,
                )
                model.load_state_dict(torch.load(args.sd + '/%s' % (model_name)))
                                                                 
            
            test_loss, test_acc, test_f1 = test(
                                                model=model,
                                                tokenizer=bert_tokenizer,
                                                test_dl=test_dl,
                                                batch_size=batch_size,
                                                sort_batch=False,
            )
            
            test_results = dict()
            test_results['test_loss'] = test_loss
            test_results['test_acc'] = test_acc
            test_results['test_f1'] = test_f1
            
            with open('./results_test/' + model_name + '.json', 'w') as json_file:
                json.dump(test_results, json_file)