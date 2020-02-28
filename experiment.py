import argparse
import datetime
import json
import os
import random
import re
import torch 
import transformers

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from collections import Counter, defaultdict
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertForQuestionAnswering
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
    parser.add_argument('--n_evals', type=str, default='multiple_per_epoch',
            help='Define number of evaluations during training. If "multiple_per_epoch", ten evals per epoch. If "one_per_epoch", once after a training epoch.')
    parser.add_argument('--sbj_classification', action='store_true',
            help='If provided, perform subjectivity classification instead of QA.')
    parser.add_argument('--multitask', action='store_true',
            help='If provided, MTL instead of STL setting.')
    parser.add_argument('--batches', type=str, default='normal',
            help='If "alternating", auxiliary task will be conditioned on question-answer sequence; elif "normal" input is question-review sequence as usual. Only necessary, if MTL setting.')
    parser.add_argument('--adversarial', type=str, default=None,
            help='If provided, adversarial instead of classic training. Only necessary, if MTL setting. Specify which adversarial version.')
    parser.add_argument('--n_aux_tasks', type=int, default=None,
            help='Define number of auxiliary tasks QA model should perform during training. Only necessary, if MTL setting.')
    parser.add_argument('--task_sampling', type=str, default='uniform',
            help='If "uniform", main and auxiliary tasks will be sampled uniformly. If "oversampling", main task will be oversampled. Only necessary, if MTL setting.')
    parser.add_argument('--encoder', action='store_true',
            help='If provided, use BiLSTM encoder to compute temporal dependencies before returning feature representations to linear output layers.')
    parser.add_argument('--highway_connection', action='store_true',
            help='If provided, put Highway connection in between BERT OR BiLSTM encoder and fc linear output head.')
    parser.add_argument('--decoder', action='store_true',
            help='If provided, put BiLSTM or BiGRU in between Highway bridge and fc linear output layers; requires BiLSTM-Encoder (and Highway bridge).')
    parser.add_argument('--bert_weights', type=str, default='not_finetuned',
            help='If finetuned, load pre-trained weights from DistilBERT model fine-tuned on SQuAD; else, load pre-trained weights from DistilBERT base model.')
    parser.add_argument('--batch_size', type=int, default=16,
            help='Define mini-batch size.')
    parser.add_argument('--n_epochs', type=int, default=3,
            help='Set number of epochs model should be fine-tuned for. If we fine-tune on SubjQA or combined, an additional epoch will be added.')
    parser.add_argument('--optim', type=str, default='AdamW',
            help='Define optimizer. Must be one of {AdamW, Adam}.')
    parser.add_argument('--sd', type=str, default='saved_models',
            help='Set model save directory for QA model.')
    parser.add_argument('--not_finetuned', action='store_true',
            help='If provided, test pre-trained DistilBERT model on SubjQA (no prior task-specific fine-tuning); only possible in test version.')
    args = parser.parse_args()
    
    # check whether arg.parser works correctly
    print(args)
    print()
    
    # move model and tensors to GPU, if GPU is available (device must be defined)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device == "cuda":
        # set cuda random seeds
        torch.cuda.manual_seed_all(42)

    # set some crucial hyperparameters
    max_seq_length = 512 # BERT cannot deal with sequences, where T > 512
    doc_stride = 200
    max_query_length = 100
    batch_size = args.batch_size
    sort_batch = True if args.encoder else False
    
    # create list of all review / paragraph domains in dataset(s)
    domains = ['books', 'tripadvisor', 'grocery', 'electronics', 'movies', 'restaurants', 'wikipedia']
    
    domains = domains[:-1] if args.finetuning == 'SubjQA' else domains
   
    qa_types = ['obj', 'sbj']
    datasets = ['SQuAD', 'SubjQA'] # not sure, whether this auxiliary task is actually useful

    # define, whether we should inform model about question or answer type
    qa_type = 'question'
    
    # create domain_to_idx, qa_type_to_idx and dataset_to_idx mappings (necessary for auxiliary tasks)
    idx_to_domains = idx_to_class(domains)
    domain_to_idx = class_to_idx(domains)
    domain_weights = None
    
    idx_to_qa_types = idx_to_class(qa_types)
    qa_type_to_idx = class_to_idx(qa_types)
    qa_type_weights = None
    
    idx_to_dataset = idx_to_class(datasets)
    dataset_to_idx = class_to_idx(datasets)
    dataset_weights = None

    n_domain_labels = None if args.finetuning == 'SQuAD' else len(domains)
    
    # NOTE: we use pre-trained cased model since both BERT and DistilBERT cased models perform significantly better on SQuAD than uncased versions     
    bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
    
    if args.bert_weights == 'not_finetuned':
        pretrained_weights = 'distilbert-base-cased'
        freeze_bert = False

    elif arg.bert_weights == 'finetuned' or args.sbj_classification:
        pretrained_weights = 'distilbert-base-cased-distilled-squad'
        freeze_bert = True

    if args.sbj_classification:

        assert not args.multitask
        assert isinstance(args.n_aux_tasks, type(None))
        assert isinstance(args.adversarial, type(None))

    dataset = args.finetuning
    encoding = 'recurrent' if args.encoder else 'linear'
    highway = 'Highway' if args.highway_connection else ''
    decoder = 'BiLSTM' if args.decoder else ''
    train_method = 'multitask' + '_' + str(args.n_aux_tasks) if args.multitask else 'singletask'
    eval_setup = args.n_evals
    batch_presentation = args.batches if args.multitask else ''

    if isinstance(args.adversarial, type(None)):
        training = 'classic'
    else:
        training = args.adversarial if args.adversarial == 'GRL' else 'adv' + args.adversarial

    model_name = 'DistilBERT' + '_' + encoding + '_' + highway + '_' + decoder + '_' + train_method + '_' + batch_presentation + '_' + training + '_' + dataset + '_' + eval_setup
    model_name = model_name.lower()
    
    if args.version == 'train':
        
        if args.finetuning == 'SubjQA':
        
            subjqa_data_train_df, hidden_domain_idx_train = get_data(
                                                                     source='/SubjQA/',
                                                                     split='/train',
                                                                     domain='all',
            )

            subjqa_data_dev_df, hidden_domain_idx_dev = get_data(
                                                                 source='/SubjQA/',
                                                                 split='/dev',
                                                                 domain='all',
            )
            
            # convert pd.DataFrames into list of dictionaries (as many dicts as examples)
            subjqa_data_train = convert_df_to_dict(
                                                   subjqa_data_train_df,
                                                   hidden_domain_indexes=hidden_domain_idx_train,
                                                   split='train',
            )
            subjqa_data_dev = convert_df_to_dict(
                                                 subjqa_data_dev_df,
                                                 hidden_domain_indexes=hidden_domain_idx_dev,
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

            np.random.shuffle(subjqa_features_train)
            
            subjqa_tensor_dataset_train = create_tensor_dataset(subjqa_features_train)

            subjqa_tensor_dataset_dev = create_tensor_dataset(subjqa_features_dev)

            train_dl = BatchGenerator(
                                      dataset=subjqa_tensor_dataset_train,
                                      batch_size=batch_size,
                                      sort_batch=sort_batch,
            )

            val_dl = BatchGenerator(
                                    dataset=subjqa_tensor_dataset_dev,
                                    batch_size=batch_size,
                                    sort_batch=sort_batch,
            )

            n_steps = len(train_dl)

            if args.multitask:

                if args.batches == 'alternating':
                    
                    # create different dataset for subjectivity auxiliary task (condition on question-answer sequence only instead of question-review sequence)
                    subjqa_tensor_dataset_train_aux_sbj = create_tensor_dataset(subjqa_features_train, aux_sbj_batch=True)

                    train_dl_sbj = BatchGenerator(
                                                  dataset=subjqa_tensor_dataset_train_aux_sbj,
                                                  batch_size=batch_size,
                                                  sort_batch=sort_batch,
                                                  )
                    
                    train_dl = list(zip(train_dl, train_dl_sbj))

                assert isinstance(args.n_aux_tasks, int), 'If MTL, number auf auxiliary tasks must be defined'
                if args.n_aux_tasks == 2:
                    subjqa_domains = [f.domain for f in subjqa_features_train]
                    domain_weights = get_class_weights(
                                                       subjqa_classes=subjqa_domains,
                                                       idx_to_class=idx_to_domains,
                ) 
                
                subjqa_q_types = [f.q_sbj for f in subjqa_features_train]                 
                subjqa_a_types = [f.a_sbj for f in subjqa_features_train]
                
                q_type_weights = get_class_weights(
                                                   subjqa_classes=subjqa_q_types,
                                                   idx_to_class=idx_to_qa_types,
                                                   binary=True,
                                                   qa_type='questions',
                )

                a_type_weights = get_class_weights(
                                                  subjqa_classes=subjqa_a_types,
                                                  idx_to_class=idx_to_qa_types,
                                                  binary=True,
                                                  qa_type='answers',
                )


                qa_type_weights = torch.stack((a_type_weights, q_type_weights))

            

            elif args.sbj_classification:

                if args.batches == 'alternating':

                    # create different dataset for subjectivity auxiliary task (condition on question-answer sequence only instead of question-review sequence)
                    subjqa_tensor_dataset_train_aux_sbj = create_tensor_dataset(subjqa_features_train, aux_sbj_batch=True)

                    train_dl = BatchGenerator(
                                              dataset=subjqa_tensor_dataset_train_aux_sbj,
                                              batch_size=batch_size,
                                              sort_batch=sort_batch,
                                              )
                    
                    subjqa_tensor_dataset_dev_aux_sbj = create_tensor_dataset(subjqa_features_dev, aux_sbj_batch=True)

                    val_dl = BatchGenerator(
                                           dataset=subjqa_tensor_dataset_dev_aux_sbj,
                                           batch_size=batch_size,
                                           sort_batch=sort_batch,
                                          )


                subjqa_q_types = [f.q_sbj for f in subjqa_features_train]                 
                subjqa_a_types = [f.a_sbj for f in subjqa_features_train]
                
                q_type_weights = get_class_weights(
                                                   subjqa_classes=subjqa_q_types,
                                                   idx_to_class=idx_to_qa_types,
                                                   binary=True,
                                                   qa_type='questions',
                )

                a_type_weights = get_class_weights(
                                                  subjqa_classes=subjqa_a_types,
                                                  idx_to_class=idx_to_qa_types,
                                                  binary=True,
                                                  qa_type='answers',
                )


                qa_type_weights = torch.stack((a_type_weights, q_type_weights)) 


        elif args.finetuning == 'SQuAD':
            
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
                                                                doc_stride=doc_stride,
                                                                max_query_length=max_query_length,
                                                                is_training=True,
                                                                domain_to_idx=domain_to_idx,
                                                                dataset_to_idx=dataset_to_idx,
            )

            squad_features_dev = convert_examples_to_features(
                                                             squad_examples_dev, 
                                                             bert_tokenizer,
                                                             max_seq_length=max_seq_length,
                                                             doc_stride=doc_stride,
                                                             max_query_length=max_query_length,
                                                             is_training=True,
                                                             domain_to_idx=domain_to_idx,
                                                             dataset_to_idx=dataset_to_idx,
            )

            np.random.shuffle(squad_features_train)
            
            squad_tensor_dataset_train = create_tensor_dataset(squad_features_train)

            squad_tensor_dataset_dev = create_tensor_dataset(squad_features_dev)

            train_dl = BatchGenerator(
                                      dataset=squad_tensor_dataset_train,
                                      batch_size=batch_size,
                                      sort_batch=sort_batch,
            )

            val_dl = BatchGenerator(
                                    dataset=squad_tensor_dataset_dev,
                                    batch_size=batch_size,
                                    sort_batch=sort_batch,
            )

            n_steps = len(train_dl)
            
        elif args.finetuning == 'combined':
             
            # load SubjQA data 
            subjqa_data_train_df, hidden_domain_idx_train = get_data(
                                                                     source='/SubjQA/',
                                                                     split='/train',
                                                                     domain='all',
            )

            subjqa_data_dev_df, hidden_domain_idx_dev = get_data(
                                                                 source='/SubjQA/',
                                                                 split='/dev',
                                                                 domain='all',
            )
            
            # convert pd.DataFrames into list of dictionaries (as many dicts as examples)
            subjqa_data_train = convert_df_to_dict(
                                                   subjqa_data_train_df,
                                                   hidden_domain_indexes=hidden_domain_idx_train,
                                                   split='train',
            )
            subjqa_data_dev = convert_df_to_dict(
                                                 subjqa_data_dev_df,
                                                 hidden_domain_indexes=hidden_domain_idx_dev,
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
            np.random.shuffle(subjqa_features_train)
            
            # load SQuAD data
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
                                                                doc_stride=doc_stride,
                                                                max_query_length=max_query_length,
                                                                is_training=True,
                                                                domain_to_idx=domain_to_idx,
                                                                dataset_to_idx=dataset_to_idx,
            )

            squad_features_dev = convert_examples_to_features(
                                                             squad_examples_dev, 
                                                             bert_tokenizer,
                                                             max_seq_length=max_seq_length,
                                                             doc_stride=doc_stride,
                                                             max_query_length=max_query_length,
                                                             is_training=True,
                                                             domain_to_idx=domain_to_idx,
                                                             dataset_to_idx=dataset_to_idx,
            )
            np.random.shuffle(squad_features_train)

            squad_features_train.extend(subjqa_features_train)
            squad_features_dev.extend(subjqa_features_dev)

            combined_features_train = squad_features_train
            combined_features_dev = squad_features_dev

            np.random.shuffle(combined_features_train)
            
            combined_tensor_dataset_train = create_tensor_dataset(combined_features_train)
            
            combined_tensor_dataset_dev = create_tensor_dataset(combined_features_dev)

            train_dl = BatchGenerator(
                                      dataset=combined_tensor_dataset_train,
                                      batch_size=batch_size,
                                      sort_batch=sort_batch,
            )

            val_dl = BatchGenerator(
                                    dataset=combined_tensor_dataset_dev,
                                    batch_size=batch_size,
                                    sort_batch=sort_batch,
            )

            n_steps = len(train_dl)

            if args.multitask:

                if args.batches == 'alternating':

                    # create different dataset for subjectivity auxiliary task (condition on question-answer sequence only instead of question-review sequence)
                    combined_tensor_dataset_train_aux_sbj = create_tensor_dataset(combined_features_train, aux_sbj_batch=True)

                    train_dl_sbj = BatchGenerator(
                                                  dataset=combined_tensor_dataset_train_aux_sbj,
                                                  batch_size=batch_size,
                                                  sort_batch=sort_batch,
                                                  )
                    
                    train_dl = list(zip(train_dl, train_dl_sbj))

                   
                assert isinstance(args.n_aux_tasks, int), 'If MTL, number auf auxiliary tasks must be defined'
                
                if args.n_aux_tasks == 2:
                    
                    squad_domains = [f.domain for f in squad_features_train]
                    subjqa_domains = [f.domain for f in subjqa_features_train]
                    domain_weights = get_class_weights(
                                                       subjqa_classes=subjqa_domains,
                                                       idx_to_class=idx_to_domains,
                                                       squad_classes=squad_domains,
                ) 
                
                subjqa_q_types = [f.q_sbj for f in subjqa_features_train] 
                squad_q_types = [f.q_sbj for f in squad_features_train] 
                
                subjqa_a_types = [f.a_sbj for f in subjqa_features_train]
                squad_a_types = [f.a_sbj for f in squad_features_train]
                
                q_type_weights = get_class_weights(
                                                   subjqa_classes=subjqa_q_types,
                                                   idx_to_class=idx_to_qa_types,
                                                   squad_classes=squad_q_types,
                                                   binary=True,
                                                   qa_type='questions',
                )

                a_type_weights = get_class_weights(
                                                  subjqa_classes=subjqa_a_types,
                                                  idx_to_class=idx_to_qa_types,
                                                  squad_classes=squad_a_types,
                                                  binary=True,
                                                  qa_type='answers',
                )

                qa_type_weights = torch.stack((a_type_weights, q_type_weights))

            elif args.sbj_classification:

                if args.batches == 'alternating':

                    # create different dataset for subjectivity auxiliary task (condition on question-answer sequence only instead of question-review sequence)
                    combined_tensor_dataset_train_aux_sbj = create_tensor_dataset(combined_features_train, aux_sbj_batch=True)

                    train_dl = BatchGenerator(
                                              dataset=combined_tensor_dataset_train_aux_sbj,
                                              batch_size=batch_size,
                                              sort_batch=sort_batch,
                                              )
                    
                    combined_tensor_dataset_dev_aux_sbj = create_tensor_dataset(combined_features_dev, aux_sbj_batch=True)

                    val_dl = BatchGenerator(
                                           dataset=combined_tensor_dataset_dev_aux_sbj,
                                           batch_size=batch_size,
                                           sort_batch=sort_batch,
                                          )

                subjqa_q_types = [f.q_sbj for f in subjqa_features_train] 
                squad_q_types = [f.q_sbj for f in squad_features_train] 
                
                subjqa_a_types = [f.a_sbj for f in subjqa_features_train]
                squad_a_types = [f.a_sbj for f in squad_features_train]
                
                q_type_weights = get_class_weights(
                                                   subjqa_classes=subjqa_q_types,
                                                   idx_to_class=idx_to_qa_types,
                                                   squad_classes=squad_q_types,
                                                   binary=True,
                                                   qa_type='questions',
                )

                a_type_weights = get_class_weights(
                                                  subjqa_classes=subjqa_a_types,
                                                  idx_to_class=idx_to_qa_types,
                                                  squad_classes=squad_a_types,
                                                  binary=True,
                                                  qa_type='answers',
                )

                qa_type_weights = torch.stack((a_type_weights, q_type_weights))

        # initialise QA model
        model = DistilBertForQA.from_pretrained(
                                          pretrained_weights,
                                          max_seq_length = max_seq_length,
                                          encoder = True if encoding == 'recurrent' else False,
                                          highway_connection = args.highway_connection,
                                          decoder = args.decoder,
                                          multitask = args.multitask,
                                          adversarial = True if args.adversarial == 'GRL' else False,
                                          n_aux_tasks = args.n_aux_tasks,
                                          n_domain_labels = n_domain_labels,
        )

        # set model to device
        model.to(device)

        hypers = {
                  "lr_adam": 5e-5,
                  "warmup_steps": 100 if args.finetuning == 'combined' else 50,
                  "max_grad_norm": 5,
        }

        hypers["task"] = 'Sbj_Classification' if args.sbj_classification else 'QA'
        hypers["n_epochs"] = args.n_epochs
        hypers["n_steps"] = n_steps
        hypers["n_evals"] = args.n_evals
        hypers["batch_presentation"] = args.batches
        hypers["task_sampling"] = args.task_sampling

        if args.n_evals == 'multiple_per_epoch':
            hypers["n_evals_per_epoch"] = 10 # number of times we evaluate model on dev set per epoch (not necessary, if we just evaluate once after an epoch)

        hypers["freeze_bert"] = freeze_bert
        hypers["pretrained_model"] = 'distilbert'
        hypers["optim"] = args.optim
        hypers["model_dir"] = args.sd
        hypers["model_name"] = model_name
        hypers["qa_type"] = qa_type
        hypers["dataset"] = args.finetuning
        
        if args.optim == 'AdamW':
            
            optimizer = AdamW(
                              model.parameters(), 
                              lr=hypers['lr_adam'], 
                              correct_bias=True,
            )
            
            t_total = n_steps * hypers['n_epochs'] # total number of training steps (i.e., step = iteration)
            
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
        
        else:
            raise ValueError("Optimizer must be one of {AdamW, Adam}.")

        train_results  = dict()
        if isinstance(args.n_aux_tasks, type(None)):           
            batch_losses, batch_accs_qa, batch_f1s_qa, val_losses, val_accs, val_f1s, model = train(
                                                                                                    model=model,
                                                                                                    tokenizer=bert_tokenizer,
                                                                                                    train_dl=train_dl,
                                                                                                    val_dl=val_dl,
                                                                                                    batch_size=batch_size,
                                                                                                    n_aux_tasks=args.n_aux_tasks,
                                                                                                    args=hypers,
                                                                                                    optimizer=optimizer,
                                                                                                    scheduler=scheduler,
                                                                                                    early_stopping=True,
                                                                                                    qa_type_weights=qa_type_weights,
                                                                                                    domain_weights=domain_weights,
                                                                                                    adversarial_simple=True if args.adversarial == 'simple' else False,
            )

        elif args.n_aux_tasks == 1:           
            batch_losses, batch_accs_qa, batch_f1s_qa, batch_accs_sbj, batch_f1s_sbj, val_losses, val_accs, val_f1s, model = train(
                                                                                                                                    model=model,
                                                                                                                                    tokenizer=bert_tokenizer,
                                                                                                                                    train_dl=train_dl,
                                                                                                                                    val_dl=val_dl,
                                                                                                                                    batch_size=batch_size,
                                                                                                                                    n_aux_tasks=args.n_aux_tasks,
                                                                                                                                    args=hypers,
                                                                                                                                    optimizer=optimizer,
                                                                                                                                    scheduler=scheduler,
                                                                                                                                    early_stopping=True,
                                                                                                                                    qa_type_weights=qa_type_weights,
                                                                                                                                    domain_weights=domain_weights,
                                                                                                                                    adversarial_simple=True if args.adversarial == 'simple' else False,
            )

            train_results['batch_accs_sbj'] = batch_accs_sbj
            train_results['batch_f1s_sbj'] = batch_f1s_sbj
        
        elif args.n_aux_tasks == 2:           
            batch_losses, batch_accs_qa, batch_f1s_qa, batch_accs_sbj, batch_f1s_sbj, batch_accs_domain, batch_f1s_domain, val_losses, val_accs, val_f1s, model = train(
                                                                                                                                                                        model=model,
                                                                                                                                                                        tokenizer=bert_tokenizer,
                                                                                                                                                                        train_dl=train_dl,
                                                                                                                                                                        val_dl=val_dl,
                                                                                                                                                                        batch_size=batch_size,
                                                                                                                                                                        n_aux_tasks=args.n_aux_tasks,
                                                                                                                                                                        args=hypers,
                                                                                                                                                                        optimizer=optimizer,
                                                                                                                                                                        scheduler=scheduler,
                                                                                                                                                                        early_stopping=True,
                                                                                                                                                                        qa_type_weights=qa_type_weights,
                                                                                                                                                                        domain_weights=domain_weights,
                                                                                                                                                                        adversarial_simple=True if args.adversarial == 'simple' else False,
            )

            train_results['batch_accs_sbj'] = batch_accs_sbj
            train_results['batch_f1s_sbj'] = batch_f1s_sbj
            train_results['batch_accs_domain'] = batch_accs_domain
            train_results['batch_f1s_domain'] = batch_f1s_domain        
            
        train_results['batch_losses'] = batch_losses
        train_results['batch_accs_qa'] = batch_accs_qa
        train_results['batch_f1s_qa'] = batch_f1s_qa
        train_results['val_losses'] = val_losses
        train_results['val_accs'] = val_accs
        train_results['val_f1s'] = val_f1s

        with open('./results_train/' + model_name + '.json', 'w') as json_file:
            json.dump(train_results, json_file)
        
    # we always test on SubjQA (TODO: evaluate model on both entire test data set and individual review domains)
    elif args.version == 'test':
        
            subjqa_data_test_df, hidden_domain_idx_test = get_data(
                                                                   source='/SubjQA/',
                                                                   split='/test',
                                                                   domain='all',
            )
            
            subjqa_data_test = convert_df_to_dict(
                                                  subjqa_data_test_df,
                                                  hidden_domain_indexes=hidden_domain_idx_test,
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
            
            subjqa_tensor_dataset_test = create_tensor_dataset(subjqa_features_test)  
            
            test_dl = BatchGenerator(
                                    dataset=subjqa_tensor_dataset_test,
                                    batch_size=batch_size,
                                    sort_batch=sort_batch,
                                    )
            
            if args.not_finetuned:
                # test (simple) BERT-QA-model fine-tuned on SQuAD without (prior) task-specific fine-tuning on SubjQA
                model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad')
                model_name = 'distilbert_pretrained_squad_no_fine_tuning'
                # set model to device
                model.to(device)
            else:
                model = DistilBertForQA.from_pretrained(
                                                        pretrained_weights,
                                                        max_seq_length = max_seq_length,
                                                        encoder = True if encoding == 'recurrent' else False,
                                                        highway_connection = args.highway_connection,
                                                        decoder = args.decoder,
                                                        multitask = args.multitask,
                                                        adversarial = True if args.adversarial == 'GRL' else False,
                                                        n_aux_tasks = args.n_aux_tasks,
                                                        n_domain_labels = n_domain_labels,
                )
                # load fine-tuned model
                model.load_state_dict(torch.load(args.sd + '/%s' % (model_name)))
                # set model to device
                model.to(device)
                                                                 
            
            test_loss, test_acc, test_f1 = test(
                                                model=model,
                                                tokenizer=bert_tokenizer,
                                                test_dl=test_dl,
                                                batch_size=batch_size,
                                                not_finetuned=args.not_finetuned,
            )
            
            test_results = dict()
            test_results['test_loss'] = test_loss
            test_results['test_acc'] = test_acc
            test_results['test_f1'] = test_f1
            
            with open('./results_test/' + model_name + '.json', 'w') as json_file:
                json.dump(test_results, json_file)