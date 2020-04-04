import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl 
import matplotlib.pyplot as plt
import seaborn as sns

import json
import os
import random
import re

from mpl_toolkits.mplot3d import Axes3D
from plotting import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels

np.random.seed(42)
random.seed(42)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_reps', type=str, default='per_class',
            help='If "per_class", plot hidden representations per class; if "per_token" pick random sent and plot it in latent space.')
    parser.add_argument('--task', type=str, default='QA',
            help='If QA, plot hidden reps of QA model; if "sbj_class", plot hidden reps of sbj. classification model.')
    args = parser.parse_args()
 
    # set folder and subdirectory
    folder = '/results_test/'
    subdir = '/feat_reps/'

    if args.task == 'sbj_class':
        subsubdir = '/sbj_class_multi/'
        task = 'multi_sbj'
    elif args.task == 'QA':
        if args.hidden_reps == 'per_class':
            subsubdir = '/qa_ds_agnostic/'  #'/qa_review_agnostic/'
        else:
            subsubdir = '/qa_per_token/'
        task = subsubdir.lstrip('/').rstrip('/')
        
    # create PATH
    cwd = '.'
    PATH = cwd + folder + subdir + subsubdir
    # we only want .json files
    files = [file for file in os.listdir(PATH) if file.endswith('.json')]
    
    # load files
    for f in files:
        if re.search(r'alternating', f):
            with open(PATH + f) as json_file:
                test_results_qa  = json.load(json_file)
                model_name_qa = 'tsne_question_answer' + '_' + task
                print("===============================================================")
                print("======= File loaded: {} =======".format(model_name_qa))
                print("===============================================================")
                print()
        else:
            with open(PATH + f) as json_file:
                test_results_qc  = json.load(json_file)
                model_name_qc = 'tsne_question_context' + '_' + task
                print("===============================================================")
                print("======= File loaded: {} =======".format(model_name_qc))
                print("===============================================================")
                print()
    
    # define which results are to be inspected
    test_results = test_results_qc if args.task == 'QA' else test_results_qa
    model_name = model_name_qc if args.task == 'QA' else model_name_qa
    
    ################################################################################################################
    ################ plot model's hidden states per transformer layer for each class in test set ###################
    ################################################################################################################
    
    if args.hidden_reps == 'per_class':
        # split data
        if task == 'multi_sbj':
            y_pred = np.array(test_results['predictions'])
            y_true = np.array(test_results['true_labels'])
            combined_ds = True

        elif task == 'qa_ds_agnostic':
            combined_ds = True
            y_true = np.array(test_results['sbj_labels'])

        elif task == 'qa_review_agnostic':
            combined_ds = False
            y_true = np.array(test_results['sbj_labels'])
        
        # convert hidden reps into NumPy matrices
        feat_reps_per_layer = {l: np.array(h) for l, h in test_results['feat_reps'].items()}
        
        # define vars
        labels = np.unique(y_true)
        classes = ['subjqa_obj', 'subjqa_sbj', 'squad'] 
        class_to_idx = {c: l for c, l in zip(classes, labels)}
        
        # set hyperparams
        retained_variance = .99
        rnd_state = 42
        model_name += '_' + str(retained_variance).lstrip('0.') + '_' + 'var'
        
        print("==========================================")
        print("=========== Started plotting =============")
        print("==========================================")
        print()
        plot_feat_reps_per_layer(
                         y_true=y_true,
                         feat_reps_per_layer=feat_reps_per_layer,
                         class_to_idx=class_to_idx,
                         retained_variance=retained_variance,
                         rnd_state=rnd_state,
                         model_name=model_name,
                         task=task,
                         combined_ds=combined_ds,
        )
        print("==========================================")
        print("=========== Finished plotting =============")
        print("==========================================")
        print()
    
    ################################################################################################################
    ####### Plot model's hidden states per transformer layer for each token in a randomly chosen word sequence #####
    ################################################################################################################
    
    elif args.hidden_reps == 'per_token':
        # plot random sentence for both correct and wrong prediction 
        predictions = ['correct', 'wrong']
        classes = ['context', 'question', 'answer']
        labels = np.arange(len(classes))
        class_to_idx = {c: l for c, l in zip(classes, labels)}

        # set hyperparams
        combined_ds = False
        retained_variance = .99
        rnd_state = 42
        model_name_copy = model_name[:]
        
        for k, pred in enumerate(predictions):
            print("============================================================================")
            print("================ Obtaining hidden reps for random sentence =================")
            print("============================================================================")
            print()
            feat_reps_per_layer, token_labels, rnd_sent = get_random_sent_feat_reps(test_results, pred)
            
            if k == 0:
                model_name += '_' + str(retained_variance).lstrip('0.') + '_' + 'var' + '_' + pred
            else:
                model_name = model_name_copy + '_' + str(retained_variance).lstrip('0.') + '_' + 'var' + '_' + pred
            
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
                                     model_name=model_name,
                                     task=task,
                                     combined_ds=combined_ds,
                                     plot_qa=True,
                                     sent_pair=rnd_sent,
            )
            print("================================================================")
            print("=========== Finished plotting: {} prediction =============".format(pred))
            print("================================================================")
            print()