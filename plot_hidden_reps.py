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
    predictions = ['correct_answerable', 'wrong_answerable', 'correct_answerable', 'wrong_answerable', 'correct_answerable', 'wrong_answerable']
    classes = ['context', 'question', 'answer']
    labels = np.arange(len(classes))
    class_to_idx = {c: l for c, l in zip(classes, labels)}

    #set hyperparams
    combined_ds = False
    retained_variance = .99
    rnd_state = 42
    rnd_seed = 42
    file_name_c = file_name[:]
    
    for k, pred in enumerate(predictions):

        if k > 0 and k % 2 == 0:
            rnd_seed += 1
        
        feat_reps_per_layer, token_labels, rnd_sent = get_random_sent_feat_reps(results, pred, rnd_seed)
        
        file_name = file_name_c + '_' + str(retained_variance).lstrip('0.') + '_' + 'var' + '_' + pred + '_' + str(k)

        """
        if k == 0:
            model_name += '_' + str(retained_variance).lstrip('0.') + '_' + 'var' + '_' + pred
        else:
            model_name = model_name_copy + '_' + str(retained_variance).lstrip('0.') + '_' + 'var' + '_' + pred
            break
        """
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
                                 combined_ds=combined_ds,
                                 plot_qa=True,
                                 sent_pair=rnd_sent,

        )
        print("================================================================")
        print("=========== Finished plotting: {} prediction =============".format(pred))
        print("================================================================")
        print()
