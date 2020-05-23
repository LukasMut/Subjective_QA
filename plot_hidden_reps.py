import argparse
import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt

import json
import os
import random
import re

from eval_squad import compute_exact
from plotting import *

#set random seeds to reproduce results
np.random.seed(42)
random.seed(42)

def get_hidden_reps(
                    source:str,
                    version:str,
):    
    #set folder and subdirectories
    folder = '/results_test/'
    subdir = '/feat_reps/'
    subsubdir = '/qa_per_token/'

    #create PATH
    cwd = '.'
    PATH = cwd + folder + subdir + subsubdir
    PATH += '/bert_finetuned_subjqa/' if source == 'SubjQA' else '/bert_finetuned_squad/'
    PATH += '/dev/' if version == 'train' else '/test/'

    if not os.path.exists(PATH):
        os.makedirs(PATH)
        raise FileNotFoundError('PATH was not correctly defined. Move files to PATH before executing script again.')

    #we want to exclusively capture .json files
    files = [file for file in os.listdir(PATH) if file.endswith('.json')]
    f = files.pop()

    #load hidden representations into memory
    with open(PATH + f, encoding="utf-8") as json_file:
        results = json.load(json_file)
        file_name = 'hidden_reps' + '_' + version
        print()
        print("===============================================================")
        print("======= File loaded: {} =======".format(file_name))
        print("===============================================================")
        print()

    return results, file_name

#################################################################        
##### OBTAIN FEAT REPS ON TOKEN LEVEL FOR RANDOM SENTENCE #######
################################################################

def get_random_sent_hidden_reps(
                              test_results:dict,
                              prediction:str,
                              rnd_seed:int,
):
    #unpack dev/test results
    pred_answers = test_results['predicted_answers']
    true_answers = test_results['true_answers']
    true_start_pos = test_results['true_start_pos']
    true_end_pos = test_results['true_end_pos']
    sent_pairs = test_results['sent_pairs']
    feat_reps = test_results['feat_reps']

    #[CLS] token id
    cls_tok_id = 0
    
    if prediction == 'correct':
        #indices for correct predictions
        indices = np.array([i for i, pred_ans in enumerate(pred_answers) if compute_exact(true_answers[i], pred_ans) and true_answers[i].strip() != '[CLS]' and len(true_answers[i].strip().split()) > 1])     
    else:
        #indices for wrong predictions
        indices = np.array([i for i, pred_ans in enumerate(pred_answers) if not compute_exact(true_answers[i], pred_ans) and true_answers[i].strip() != '[CLS]' and len(true_answers[i].strip().split()) > 1])

    #set random seed to reproduce plots
    np.random.seed(rnd_seed)
    #get random idx
    rnd_sent_idx = np.random.choice(indices)
    #get random sent according to random idx
    rnd_sent = sent_pairs[rnd_sent_idx]
    #convert sent into list
    rnd_sent = rnd_sent.split()
    #get sep idx (NOTE: .index() returns first occurrence of element in list)
    sep_idx = rnd_sent.index('[SEP]')
    #get question indices
    q_indices = np.arange(1, sep_idx)
    
    print("=============================================================")
    print("===== Question: {} =====".format(' '.join(rnd_sent[q_indices[0]:q_indices[-1]+1])))
    print("=============================================================")
    print()
    
    #get answer indices
    a_indices = np.arange(true_start_pos[rnd_sent_idx], true_end_pos[rnd_sent_idx] + 1)
    
    print("=============================================================")
    print("===== True answer: {} =====".format(' '.join(rnd_sent[a_indices[0]:a_indices[-1]+1])))
    print("=============================================================")
    print()

    if re.search(r'wrong', prediction):
      print("=============================================================")
      print("===== Predicted answer: {} =====".format(pred_answers[rnd_sent_idx]))
      print("=============================================================")
      print()

    #create list of special token indices
    special_tok_indices = [cls_tok_id, sep_idx, len(rnd_sent)-1]

    try:
      if a_indices == np.array([cls_tok_id]):
          special_tok_indices.pop(special_tok_indices.index(cls_tok_id))
    except ValueError:
      pass
    
    special_tok_indices = np.array(special_tok_indices)
    
    #extract feat reps for random sent on token level and convert to NumPy
    print("============================================================================")
    print("================ Obtaining hidden reps for random sentence =================")
    print("============================================================================")
    print()
    feat_reps_per_layer = {l: np.array(hiddens)[rnd_sent_idx] for l, hiddens in feat_reps.items()}
    
    # create synthetic labels for token sequence
    T = len(rnd_sent)
    token_labels = np.zeros(T, dtype=int)
    
    for i in range(T):
        if i in special_tok_indices:
            token_labels[i] += 99
        elif i in q_indices:
            token_labels[i] += 1
        elif i in a_indices:
            token_labels[i] += 2
    
    return feat_reps_per_layer, token_labels, rnd_sent
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='SQuAD',
            help='Must be one of {SQuAD, SubjQA}')
    parser.add_argument('--version', type=str, default='train',
            help='Must be one of {train, test}')
    args = parser.parse_args()

    results, file_name = get_hidden_reps(source=args.source, version=args.version)
    
    ################################################################################################################
    ####### Plot model's hidden states per transformer layer for each token in a randomly chosen word sequence #####
    ################################################################################################################
    
    #plot random sentence for both correct and incorrect (answer span) predictions 
    predictions = ['correct' if i % 2 == 0 else 'wrong' for i in range(10)]
    classes = ['context', 'question', 'answer']
    labels = np.arange(len(classes))
    class_to_idx = {c: l for c, l in zip(classes, labels)}

    #set hyperparams
    retained_variance = .99
    rnd_state = 42
    rnd_seed = rnd_state
    file_name_c = file_name[:]
    
    for k, pred in enumerate(predictions):

        if k > 0 and k % 2 == 0:
            rnd_seed += 1
        
        feat_reps_per_layer, token_labels, rnd_sent = get_random_sent_hidden_reps(results, pred, rnd_seed)
        
        file_name = file_name_c + '_' + str(retained_variance).lstrip('0.') + '_' + 'var' + '_' + pred + '_' + str(k)

        print("================================================================")
        print("=========== Started plotting: {} prediction =============".format(pred))
        print("================================================================")
        print()
        plot_feat_reps_per_layer(
                                 y_true=token_labels,
                                 feat_reps_per_layer=feat_reps_per_layer,
                                 class_to_idx=class_to_idx,
                                 retained_variance=retained_variance,
                                 rnd_state=rnd_state,
                                 file_name=file_name,
                                 source=args.source,
                                 version=args.version,
                                 combined_ds=False,
                                 plot_qa=True,
                                 sent_pair=rnd_sent,

        )
        print("================================================================")
        print("=========== Finished plotting: {} prediction =============".format(pred))
        print("================================================================")
        print()
