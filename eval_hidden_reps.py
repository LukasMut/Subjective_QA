#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = [
           'get_hidden_reps',
           'compute_ans_similarities',
           'adjust_p_values',
           'shuffle_arrays',
           'compute_similarities_across_layers',
           'evaluate_estimations_and_cosines',
           ]

import argparse
import json
import matplotlib
import os
import pickle
import re
import torch

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch.nn as nn

from collections import defaultdict, Counter
from eval_squad import compute_exact
from joblib import dump, load
from nltk.translate.bleu_score import sentence_bleu
from statsmodels.stats.multitest import multipletests
from scipy import io
from scipy.stats import entropy, f_oneway, spearmanr, ttest_ind
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.utils import shuffle
from tqdm import trange, tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam, SGD
from utils import BatchGenerator

try:
    from models.utils import to_cpu, f1, soft_to_hard, accuracy
    from models.modules.NN import *
except ImportError:
    pass

#move model and tensors to GPU, if GPU is available (device must be defined)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_hidden_reps(
                    source:str,
                    version:str,
):    
    #set folder and subdirectories
    folder = '/results_test/'
    subdir = '/feat_reps/'
    subsubdir = '/qa_per_token/'
    task = 'ans_pred'

    #create PATH
    cwd = '.'
    PATH = cwd + folder + subdir + subsubdir
    PATH += '/bert_finetuned_subjqa/' if source == 'SubjQA' else '/bert_finetuned_squad/'
    PATH += '/test/' if version == 'train' else '/dev/' #NOTE: change this back to: '/dev/' if version == 'train' else '/test/'

    if not os.path.exists(PATH):
        os.makedirs(PATH)
        raise FileNotFoundError('PATH was not correctly defined. Move files to PATH before executing script again.')

    #we want to exclusively capture .json files
    files = [file for file in os.listdir(PATH) if file.endswith('.json')]
    f = files.pop()

    #load hidden representations into memory
    with open(PATH + f, encoding="utf-8") as json_file:
        results = json.load(json_file)
        file_name = 'hidden_rep_cosines' + '_' +  task + '_' + version
        print()
        print("===============================================================")
        print("======= File loaded: {} =======".format(file_name))
        print("===============================================================")
        print()

    return results, file_name

def euclidean_dist(u, v): return np.linalg.norm(u-v) #default is L2 norm

def kl_div(p, q):
    #NOTE: probability distributions p and q must sum to 1 (normalization factor required)
    p /= np.sum(p, keepdims=True)
    q /= np.sum(q, keepdims=True)
    rel_entr = p * np.log(p/q)
    return np.sum(rel_entr)

def cosine_sim(u, v):
    num = u @ v
    denom = np.linalg.norm(u) * np.linalg.norm(v) #default is Frobenius norm (i.e., L2 norm)
    return num / denom

def compute_ans_similarities(a_hiddens:np.ndarray):
    a_dists = []
    for i, a_i in enumerate(a_hiddens):
        for j, a_j in enumerate(a_hiddens):
            #NOTE: cos sim is a symmetric dist metric plus we don't want to compute cos sim of a vector with itself (i.e., cos_sim(u, u) = 1)
            if i != j and j > i:
                a_dists.append(cosine_sim(u=a_i, v=a_j))
    return np.max(a_dists), np.min(a_dists), np.mean(a_dists), np.std(a_dists)

def adjust_p_values(
                    ans_similarities:dict,
                    alpha:float=.05,
                    adjustment:str='bonferroni',
                    ):
    uncorrected_p_vals = np.array([vals['ttest_p_val'] for l, vals in ans_similarities.items()])
    corrected_p_vals = multipletests(pvals=uncorrected_p_vals, alpha=alpha, method=adjustment, returnsorted=False)[1]
    for l, p_val in enumerate(corrected_p_vals):
        ans_similarities['Layer'+'_'+str(l+1)]['ttest_p_val'] = p_val
    return ans_similarities

def shuffle_arrays(X:np.ndarray, y:np.ndarray):
    assert X.shape[0] == y.shape[0]
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]

def create_tensor_dataset(X:np.ndarray, y:np.ndarray): return TensorDataset(torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.long))

def test(model, test_dl):
    n_iters = len(test_dl)
    test_f1 = 0
    test_acc = 0
    test_steps = 0
    i = 0
    incorrect_preds = []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(test_dl):
            batch = tuple(t.to(device) for t in batch)
            X, y = batch
            logits = model(X)
            probas = torch.sigmoid(logits)
            y_pred = soft_to_hard(probas).long()
            incorrect_preds.append(np.where(y.cpu() != y_pred)[0] + i)
            test_f1 += f1(probas=probas, y_true=y, task='binary')
            test_acc += accuracy(probas=probas, y_true=y, task='binary')
            test_steps += 1
            i += X.size(0) #batch_size
        test_f1 /= test_steps
        test_acc /= test_steps
    #NOTE: np.array().flatten() can only be applied to a matrix, iff all rows (i.e., vectors) are of equal length (same dimensionality required);
    #      use list comprehension instead to flatten a matrix whose rows are not of equal length (different dimensionalities)
    incorrect_preds = np.array([el for batch in incorrect_preds for el in batch])
    print("===================")
    print("==== Inference ====")
    print("==== F1: {} ====".format(round(test_f1, 3)))
    print("=== Acc: {} ===".format(round(test_acc, 3)))
    print("===================")
    print()
    return test_f1, test_acc, incorrect_preds

def train(
          model,
          train_dl,
          version:str,
          n_epochs:int,
          batch_size:int,
          y_weights:torch.Tensor,
):
    n_steps = len(train_dl)
    n_iters = n_steps * n_epochs
    min_n_epochs = 15 if version.lower() == 'subjqa' else 10
    assert isinstance(y_weights, torch.Tensor), 'Tensor of weights wrt model predictions is not provided'
    loss_func = nn.BCEWithLogitsLoss(pos_weight=y_weights.to(device))
    optimizer = Adam(model.parameters(), lr=1e-2, weight_decay=0.005) #L2 Norm (i.e., weight decay)
    max_grad_norm = 10 #if version.lower() == 'subjqa' else 5
    losses = []
    f1_scores = []

    for epoch in range(n_epochs):
        model.train()
        train_f1 = 0
        train_steps = 0
        train_loss = 0
        for step, batch in enumerate(train_dl):
            batch = tuple(t.to(device) for t in batch)
            X, y = batch
            optimizer.zero_grad()
            logits = model(X)
            y = y.type_as(logits)
            loss = loss_func(logits, y)
            train_f1 += f1(probas=torch.sigmoid(logits), y_true=y, task='binary')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            train_steps += 1
            train_loss += loss.item()

        losses.append(train_loss / train_steps)
        f1_scores.append(train_f1 / train_steps)

        print("=============================")
        print("======== Epoch: {} ==========".format(epoch + 1))
        print("======= Loss: {} ========".format(round(losses[-1], 3)))
        print("======= F1: {} ==========".format(round(f1_scores[-1], 3)))
        print("============================")
        print()

        if epoch >= min_n_epochs:
            if losses[-1] >= losses[-2]:
                break

    model.eval()
    return losses, f1_scores, model

def plot_cosine_boxplots(
                         a_correct_cosines_mean:np.ndarray,
                         a_incorrect_cosines_mean:np.ndarray,
                         source:str,
                         version:str,
                         layer_no:str,
                         boxplot_version:str,
):
    plt.figure(figsize=(6, 4), dpi=100)

    #set fontsize var
    lab_fontsize = 12

    ax = plt.subplot(111)

    #hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #only show ticks on the left (y-axis) and bottom (x-axis) spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    if boxplot_version == 'seaborn':
        
        sns.boxplot(
                     data=[a_correct_cosines_mean, a_incorrect_cosines_mean],
                     color='white',
                     meanline=False, #not necessary to show dotted mean line when showmeans = True (only set one of the two to True)
                     showmeans=True,
                     showcaps=True,
                     showfliers=True,
                     )

        ax.set_xticklabels(['correct', 'erroneous'])
        ax.set_xlabel('answer predictions', fontsize=lab_fontsize)
        ax.set_ylabel(r'$\cos$ sim', fontsize=lab_fontsize)

        for i, artist in enumerate(ax.artists):
            if i % 2 == 0:
                col = 'pink'
            else:
                col = 'steelblue'

            #this sets the color for the main box
            artist.set_edgecolor(col)
            
            #each box has 7 associated Line2D objects (to make the whiskers, median lines, means, fliers, etc.)
            #loop over them, and use the same colour as above (display means in black to make them more salient)
            for j in range(i*7,i*7+7):
                line = ax.lines[j]
                line.set_color('black' if j == 5 + (i*7) else col)
                line.set_mfc('black' if j == 5 + (i*7) else col)
                line.set_mec('black' if j == 5 + (i*7) else col)

    elif boxplot_version == 'matplotlib':
        
        plt.boxplot(
                    x=[a_correct_cosines_mean, a_incorrect_cosines_mean],
                    notch=True,
                    bootstrap=1000,
                    meanline=False, 
                    showmeans=True, 
                    labels=['correct', 'erroneous'],
                    )
        plt.xlabel('answer predictions', fontsize=lab_fontsize)
        plt.ylabel(r'$\cos$ sim', fontsize=lab_fontsize)

    plt.tight_layout()

    PATH = './plots/hidden_reps/cosine_distributions/' + source.lower() + '/' + version + '/' + 'boxplots' + '/'
   
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    
    plt.savefig(PATH + 'layer' + '_' + layer_no +  '_' + boxplot_version + '.png')
    plt.close()

def plot_cosine_cdfs(
                     a_correct_cosines_mean:np.ndarray,
                     a_incorrect_cosines_mean:np.ndarray,
                     source:str,
                     version:str,
                     layer_no:str,
):
    cos_correct_sorted = np.sort(a_correct_cosines_mean)
    cos_incorrect_sorted = np.sort(a_incorrect_cosines_mean)
    p_correct = np.arange(1, len(cos_correct_sorted)+1) / len(cos_correct_sorted)
    p_incorrect = np.arange(1, len(cos_incorrect_sorted)+1) / len(cos_incorrect_sorted)

    #the higher the dpi, the better is the resolution of the plot (be aware that this will increase MB of .png file -> don't set dpi too high)
    plt.figure(figsize=(6, 4), dpi=100)

    #set vars
    legend_fontsize = 8
    lab_fontsize = 10

    ax = plt.subplot(111)

    #hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #only show ticks on the left (y-axis) and bottom (x-axis) spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plt.plot(cos_correct_sorted, p_correct, label='correct answers')
    plt.plot(cos_incorrect_sorted, p_incorrect, label='wrong answers')
    plt.xlabel(r'$\cos$ sim', fontsize=lab_fontsize)
    plt.ylabel(r'$p$', fontsize=lab_fontsize)
    plt.legend(fancybox=True, shadow=True, loc='best', fontsize=legend_fontsize)
    plt.tight_layout()

    PATH = './plots/hidden_reps/cosine_distributions/' + source.lower() + '/' + version + '/' + 'cdfs' + '/'
    
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    plt.savefig(PATH + 'layer' + '_' + layer_no + '.png')
    plt.close()

def plot_cosine_distrib(
                        a_correct_cosines_mean:np.ndarray,
                        a_incorrect_cosines_mean:np.ndarray,
                        source:str,
                        version:str,
                        layer_no:str,
):
    #the higher the dpi, the better is the resolution of the plot (be aware that this will increase MB of .png file -> don't set dpi too high)
    plt.figure(figsize=(6, 4), dpi=100)

    #set vars
    legend_fontsize = 8
    lab_fontsize = 10

    ax = plt.subplot(111)

    #hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #only show ticks on the left (y-axis) and bottom (x-axis) spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
        
    sns.distplot(a_correct_cosines_mean, kde=True, norm_hist=True, label='correct answers')
    sns.distplot(a_incorrect_cosines_mean, kde=True, norm_hist=True, label='wrong answers')
    plt.xlabel(r'$\cos$ sim', fontsize=lab_fontsize)
    plt.ylabel('probability density', fontsize=lab_fontsize)
    plt.legend(fancybox=True, shadow=True, loc='best', fontsize=legend_fontsize)
    plt.tight_layout()

    PATH = './plots/hidden_reps/cosine_distributions/' + source.lower() + '/' + version + '/' + 'density_plots' + '/'
    
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    plt.savefig(PATH + 'layer' + '_' + layer_no + '.png')
    plt.close()

def compute_rel_freq(cos_sim_preds:dict):
    return {layer: {pred: {'min_std_cos':vals['min_std_cos']/vals['freq'], 'max_mean_cos':vals['max_mean_cos']/vals['freq'], 'spearman_r':np.mean(vals['spearman_r'])} for pred, vals in preds.items()} for layer, preds in cos_sim_preds.items()}

def remove_impossible_candidates(s_positions:np.ndarray, e_positions:np.ndarray):
    s_candidates, e_candidates = zip(*[(s_pos, e_pos) for s_pos, e_pos in zip(s_positions, e_positions) if s_pos < e_pos])
    return s_candidates, e_candidates

def compute_cos_sim_across_logits(
                                  hiddens:np.ndarray,
                                  s_log_probs:np.ndarray,
                                  e_log_probs:np.ndarray,
                                  cos_similarities_preds:dict,
                                  true_pred:bool,
                                  layer:str,
                                  top_k:int,
                                  ):
    assert len(s_log_probs) == len(e_log_probs)
    #sort log-probabilities in decreasing order (0 to -inf)
    s_positions = np.argsort(s_log_probs)[::-1]
    e_positions = np.argsort(e_log_probs)[::-1]
    #remove impossible answer span predictions to yield an array of possible candidate answers (i.e., remove answer spans where s_pos >= e_pos)
    s_candidates, e_candidates = remove_impossible_candidates(s_positions, e_positions)
    top_k_s_candidates = s_candidates[:top_k]
    top_k_e_candidates = e_candidates[:top_k]

    _, _, mean_cosines, std_cosines = zip(*[compute_ans_similarities(hiddens[top_k_s_candidates[i]:top_k_e_candidates[i]+1,:]) for i in range(top_k)])

    cos_similarities_preds[layer]['correct' if true_pred else 'erroneous'] = {}
    
    try:
        cos_similarities_preds[layer]['correct' if true_pred else 'erroneous']['freq'] += 1
    except KeyError:
        cos_similarities_preds[layer]['correct' if true_pred else 'erroneous']['freq'] = 1
        cos_similarities_preds[layer]['correct' if true_pred else 'erroneous']['max_mean_cos'] = 0
        cos_similarities_preds[layer]['correct' if true_pred else 'erroneous']['min_std_cos'] = 0
        cos_similarities_preds[layer]['correct' if true_pred else 'erroneous']['spearman_r'] = []

    if np.argmax(np.asarray(mean_cosines)) == 0:
        cos_similarities_preds[layer]['correct' if true_pred else 'erroneous']['max_mean_cos'] += 1

    if np.argmin(np.asarray(std_cosines)) == 0:
        cos_similarities_preds[layer]['correct' if true_pred else 'erroneous']['min_std_cos'] += 1

    cos_similarities_preds[layer]['correct' if true_pred else 'erroneous']['spearman_r'].append(spearmanr(mean_cosines, std_cosines)[0])

    return cos_similarities_preds

def interp_cos_per_layer(
                         X:np.ndarray,
                         source:str,
                         version:str,
                         layers:str,
                         w_strategy:str='cdf',
                         computation:str='concat',
                         concatenation:str='rearrange',
                         delta:float=.1,
                         y=None,
):
    """
        - interpolate cos(h_a) per layer according to CDFs (Cumulative Distribution Functions) wrt cos(h_a) *train* distribution 
        - corresponding to incorrect or erroneous predictions respectively
        - replace both mean(cos(h_a)) and std(cos(h_a)) with interpolated probability values
        - (i.e., probability values that denote how likely observed *test* cos(h_a) lies within pre-defined interval given *train* cos(h_a) CDF)
    """
    PATH = './results_hidden_reps/' + source.lower() + '/cosines'
    subdir_correct = '/correct'
    subdir_incorrect = '/incorrect'
    file_name = '/cosine_distrib' + '_' + layers + '.mat'

    cos_distrib_correct_preds = io.loadmat(PATH + subdir_correct + file_name)['out']
    cos_distrib_incorrect_preds = io.loadmat(PATH + subdir_incorrect + file_name)['out']

    if computation == 'concat':
        assert len(cos_distrib_correct_preds) == len(cos_distrib_incorrect_preds)
        L = len(cos_distrib_correct_preds)
        cdf_probas = np.zeros((X.shape[0], 2*L))      
    
    def interp_cos(
                   x:float,
                   cos:np.ndarray,
                   weighting:bool=False,
                   delta=None,
    ):
        """
            - compute P(x_i - delta < x_i < x_i + delta) (is equal to P(x_i - delta <= x_i <= x_i + delta)) => p that observed cos(h_a) lies within pre-defined interval according to CDFs
            - set endpoint flag in np.linspace() to False to yield an unbiased estimator of the CDF (equivalent to np.arange(1, len(cos)+1)/len(cos))
        """
        p = np.arange(1, len(cos)+1) / len(cos) #np.linspace(0, 1, len(cos), endpoint=False)
        cos_sorted = np.sort(cos) #sort values in ascending order
        assert np.all(np.diff(cos_sorted) >= 0), 'x-coordinate sequence xp must be passed in increasing order' #use >= 0 since some values might be equivalent (hence, > 0 will yield AssertionError)
        
        if weighting:
            return np.interp(x, cos_sorted, p) #P(cos(h_a) < x_i)
        else:
            assert isinstance(delta, float), 'cutoff value to compute interval must be provided'
            return np.interp(x+delta, cos_sorted, p) - np.interp(x-delta, cos_sorted, p) #P(cos(h_a) < x_i + delta) - P(cos(h_a) < x_i - delta) 
    
    for l, (cos_correct, cos_incorrect) in enumerate(zip(cos_distrib_correct_preds, cos_distrib_incorrect_preds)):
        #unpack mean(cos(h_a)) and std(cos(h_a)) *train* distributions
        cos_correct_means, cos_correct_stds = zip(*cos_correct)
        cos_incorrect_means, cos_incorrect_stds = zip(*cos_incorrect)
        
        if computation == 'concat':
            cdf_probas_per_layer = []

        for i in range(X.shape[0]):
            cos_mean = X[i, 2*l]
            cos_std = X[i, 2*l+1]
            
            if version == 'train' or w_strategy == 'oracle':
                assert isinstance(y, np.ndarray), 'y must be provided at train time or if w_strategy == oracle'
                p_cos_mean = interp_cos(x=cos_mean, cos=cos_correct_means if y[i] == 1 else cos_incorrect_means, delta=delta)
                p_cos_std = interp_cos(x=cos_std, cos=cos_correct_stds if y[i] == 1 else cos_incorrect_stds, delta=delta)
            else:
                #NOTE: we shall not exploit gold labels (i.e., QA model predictions) at test time
                p_cos_mean_correct = interp_cos(x=cos_mean, cos=cos_correct_means, delta=delta)
                p_cos_mean_incorrect = interp_cos(x=cos_mean, cos=cos_incorrect_means, delta=delta)
                p_cos_std_correct = interp_cos(x=cos_std, cos=cos_correct_stds, delta=delta)
                p_cos_std_incorrect = interp_cos(x=cos_std, cos=cos_incorrect_stds, delta=delta)

                dist_cos_mean_correct = abs(np.mean(cos_correct_means) - cos_mean)
                dist_cos_mean_incorrect = abs(np.mean(cos_incorrect_means) - cos_mean)
                dist_cos_std_correct = abs(np.mean(cos_correct_stds) - cos_std)
                dist_cos_std_incorrect = abs(np.mean(cos_incorrect_stds) - cos_std)

                if w_strategy == 'distance':
                    cos_mean_w_correct = 1 - dist_cos_mean_correct
                    cos_mean_w_incorrect = 1 - dist_cos_mean_incorrect
                    cos_std_w_correct = 1 - dist_cos_std_correct
                    cos_std_w_incorrect = 1 - dist_cos_std_incorrect

                elif w_strategy == 'cdf':

                    #################################################################################################################
                    ## Note that the smaller abs(P(cos(h_a) > x_i) - P(cos(h_a) < x_i)) is,                                        ##
                    ## the higher is the likelihood that x_i belongs to the respective distribution. This is a property of CDFs.   ##
                    ## Hence, we must leverage 1 - abs(P(cos(h_a) > x_i) - P(cos(h_a) < x_i)) as a scaling factor (i.e., weights)  ##
                    ## to approximate the *true* p_cdf values corresponding to the train CDFs.                                     ##
                    #################################################################################################################

                    q_cos_mean_correct = 1 - interp_cos(x=cos_mean, cos=cos_correct_means, weighting=True)
                    cdf_cos_mean_correct = interp_cos(x=cos_mean, cos=cos_correct_means, weighting=True)
                    q_cos_std_correct = 1 - interp_cos(x=cos_std, cos=cos_correct_stds, weighting=True)
                    cdf_cos_std_correct = interp_cos(x=cos_std, cos=cos_correct_stds, weighting=True)

                    q_cos_mean_incorrect = 1 - interp_cos(x=cos_mean, cos=cos_incorrect_means, weighting=True)
                    cdf_cos_mean_incorrect = interp_cos(x=cos_mean, cos=cos_incorrect_means, weighting=True)
                    q_cos_std_incorrect = 1 - interp_cos(x=cos_std, cos=cos_incorrect_stds, weighting=True)
                    cdf_cos_std_incorrect = interp_cos(x=cos_std, cos=cos_incorrect_stds, weighting=True)

                    cos_mean_w_correct = 1 - abs(q_cos_mean_correct - cdf_cos_mean_correct)
                    cos_mean_w_incorrect = 1 - abs(q_cos_mean_incorrect - cdf_cos_mean_incorrect)
                    cos_std_w_correct = 1 - abs(q_cos_std_correct - cdf_cos_std_correct)
                    cos_std_w_incorrect = 1 - abs(q_cos_std_incorrect - cdf_cos_std_incorrect)

                #weighted sum of the probabilities that *observed* cos(h_a) belongs to the distribution of correct or incorrect answer predictions respectively
                p_cos_mean = ((p_cos_mean_correct * cos_mean_w_correct) + (p_cos_mean_incorrect * cos_mean_w_incorrect)) / 2
                p_cos_std = ((p_cos_std_correct * cos_std_w_correct) + (p_cos_std_incorrect * cos_std_w_incorrect)) / 2
                
            if computation == 'weighting':
                #use p as a weighting factor for mean and std wrt cos(h_a)
                X[i, 2*l] *= p_cos_mean
                X[i, 2*l+1] *= p_cos_std
            
            elif computation == 'concat':
                cdf_probas_per_layer.append((p_cos_mean, p_cos_std))

        if computation == 'concat':
            cdf_probas[:, 2*l:2*l+2] += np.stack(zip(*cdf_probas_per_layer), axis=1)

    if computation == 'concat':
        assert isinstance(concatenation, str), 'how to concatenate feature matrices must be specified'
        if concatenation == 'rearrange':
            def rearrange_values(x:np.ndarray):
                """
                rearrange values to enhance proximity between features that are correlated;
                this might inform neural network better about likelihood that *observed* cos(h_a) belongs to one or the other class;
                """
                M = len(x)
                means = x[slice(0, M//2, 2)]
                stds = x[slice(1, M//2, 2)]
                p_means = x[slice(M//2, None, 2)]
                p_stds = x[slice(M//2+1, None, 2)]
                assert len(means) == len(stds) == len(p_means) == len(p_stds)
                rearranged_array = np.array([(mean, p_means[l], stds[l], p_stds[l]) for l, mean in enumerate(means)]).flatten()
                return rearranged_array

            X = np.hstack((X, cdf_probas))
            X = np.asarray(list(map(lambda x: rearrange_values(x), X)))

        elif concatenation == 'reversed':
            #concatenate matrix of probability scores in reversed order to increase proximity between correlated features (i.e., features that belong to the same layer)
            X = np.hstack((X, cdf_probas[:, ::-1]))
        else:
            X = np.hstack((X, cdf_probas))
    return X

def compute_n_gram_overlap(q:list, a_candidate:list):
    n_grams = list(range(1, 4))
    n_gram_overlaps = []
    def compute_unique_n_grams(sent:list, n_gram:int):
        return set(tuple(sent[i:i+n_gram]) for i in range(len(sent)) if i <= len(sent)-n_gram)
    for n_gram in n_grams:
        q_n_grams = compute_unique_n_grams(q, n_gram)
        a_n_grams = compute_unique_n_grams(a_candidate, n_gram)
        try:
            overlap_prop_q = len(q_n_grams.intersection(a_n_grams))/len(q_n_grams)
            overlap_prop_a = len(a_n_grams.intersection(q_n_grams))/len(a_n_grams)
        except:
            overlap_prop_q = float(0)
            overlap_prop_a = float(0)
        n_gram_overlaps.append((overlap_prop_q, overlap_prop_a))
    n_gram_overlaps = np.asarray(n_gram_overlaps).flatten()
    return n_gram_overlaps

def compute_baseline_features(
                              feat_reps:dict,
                              sent_pairs:list,
                              pred_indices:list,
                              s_log_probs:np.ndarray,
                              e_log_probs:np.ndarray,
                              last_layer:str='Layer_6',
                              method:str='heuristic',
):
    N = len(pred_indices)
    M = 9 #ans_length, cos_sim, bleu_score, n_gram_overlaps
    D = 768
    X = np.zeros((N, M)) if method == 'heuristic' else np.zeros((N, 2*D))
    for i, sent_pair in enumerate(sent_pairs):
        if i in pred_indices:
            sent_pair = sent_pair.strip().split()
            sep_idx = sent_pair.index('[SEP]')
            q = sent_pair[1:sep_idx]
            #extract hidden reps for question and candidate answer
            hiddens = np.asarray(feat_reps[last_layer][i])
            q_hiddens = hiddens[1:sep_idx]
            q_mean_rep = q_hiddens.mean(axis=0)
            s_pos = np.argmax(s_log_probs[i])
            e_pos = np.argmax(e_log_probs[i])
            a_hiddens = hiddens[s_pos:e_pos+1]

            try:
                a_mean_rep = a_hiddens.mean(axis=0)
            except:
                a_mean_rep = np.zeros(D)

            if method == 'heuristic':
                #compute cos sim between avg q_hidden and avg a_pred_hidden
                cos_sim = cosine_sim(q_mean_rep, a_mean_rep)
                cos_sim = 0 if np.isnan(cos_sim) else cos_sim

                #compute length of a_pred
                a_candidate = sent_pair[s_pos:e_pos+1]
                a_candidate_len = len(a_candidate)

                #compute BLEU score, where q serves as the reference and a_pred as the hypothesis (i.e., translation)
                try:
                    bleu_score = sentence_bleu([q], a_candidate)
                except ZeroDivisionError:
                    bleu_score = sentence_bleu([q], a_candidate, weights=(0.5, 0.5))
                except:
                    bleu_score = 0 

                #compute n_gram overlaps between q and a_pred
                n_gram_overlaps = compute_n_gram_overlap(q, a_candidate) if a_candidate_len > 0 else np.zeros(6)

                #concatenate features
                X[i] += np.concatenate((np.array([a_candidate_len, cos_sim, bleu_score]), n_gram_overlaps))

            else:
                #concat hiddens w.r.t. q and a_pred
                X[i] += np.concatenate((q_mean_rep, a_mean_rep))
    return X

def compute_similarities_across_layers(
                                       feat_reps:dict,
                                       true_start_pos:list,
                                       true_end_pos:list,
                                       sent_pairs:list,
                                       pred_indices:list,
                                       true_preds:np.ndarray,
                                       s_log_probs:np.ndarray,
                                       e_log_probs:np.ndarray,
                                       source:str,
                                       version:str,
                                       top_k:int=10,
                                       layers=None,
):
    retained_var = .95 #retain 90% or 95% of the hidden rep's variance
    rnd_state = 42 #set random state for reproducibility
    N = len(pred_indices)
    ans_similarities = defaultdict(dict)
    cos_similarities_preds = defaultdict(dict)    
    
    est_layers = list(range(1, 7)) if layers == 'all_layers' else list(range(4, 7))
    L = len(est_layers)
    M = 2
    X = np.zeros((N, M*L))
    j = 0 #running idx to update X_i for each l in L_est

    #initialise PCA (we need to apply PCA to remove noise from the high-dimensional feature representations)
    pca = PCA(n_components=retained_var, svd_solver='auto', random_state=rnd_state)

    if version == 'train':
        correct_preds_cosines_per_layer = []
        incorrect_preds_cosines_per_layer = []
    
    for l, hiddens_all_sents in feat_reps.items():
        correct_preds_cosines = []
        incorrect_preds_cosines = []
        layer_no = int(l.lstrip('Layer' + '_'))
        k = 0
        for i, hiddens in enumerate(hiddens_all_sents):
            if i in pred_indices:
                hiddens = np.asarray(hiddens)
                sent = sent_pairs[i].strip().split()
                sep_idx = sent.index('[SEP]')

                #remove hidden reps corresponding to special [CLS] and [SEP] tokens
                #hiddens = np.vstack((hiddens[1:sep_idx], hiddens[sep_idx+1:-1])) 
                
                #transform hidden reps with PCA
                hiddens = pca.fit_transform(hiddens)

                if layer_no == 1 and i == pred_indices[0]:
                    print("==============================================================")
                    print("=== Number of components in transformed hidden reps: {} ===".format(hiddens.shape[1]))
                    print("==============================================================")
                    print()

                elif layer_no > 3:
                #TODO: the if statement below is just a work-around for now (must be fixed properly later)
                    if source.lower() == 'squad':
                        cos_similarities_preds = compute_cos_sim_across_logits(
                                                                               hiddens=hiddens,
                                                                               s_log_probs=s_log_probs[i],
                                                                               e_log_probs=e_log_probs[i],
                                                                               cos_similarities_preds=cos_similarities_preds,
                                                                               true_pred=bool(true_preds[pred_indices == i]),
                                                                               layer=l,
                                                                               top_k=top_k,
                                                                               )

                #extract hidden reps for answer span
                #a_hiddens = hiddens[true_start_pos[i]-2:true_end_pos[i]-1] #move ans span indices two positions to the left (accounting for the removal of [CLS] and [SEP])
                a_hiddens = hiddens[true_start_pos[i]:true_end_pos[i]+1]

                #compute cos(h_a)
                _, _, a_mean_cos, a_std_cos = compute_ans_similarities(a_hiddens)

                if layer_no in est_layers:
                    X[k, M*j:M*j+M] += np.array([a_mean_cos, a_std_cos])
            
                if true_preds[pred_indices == i] == 1:
                    correct_preds_cosines.append((a_mean_cos, a_std_cos))
                else:
                    incorrect_preds_cosines.append((a_mean_cos, a_std_cos))

                k += 1

        if version == 'train':
            #store mean cos(h_a) and std cos(h_a) distributions for every transformer layer to compute *train* CDFs
            if layer_no in est_layers:
                correct_preds_cosines_per_layer.append(correct_preds_cosines)
                incorrect_preds_cosines_per_layer.append(incorrect_preds_cosines)

        #unpack means and stds wrt cos(h_a)
        a_correct_cosines_mean, a_correct_cosines_std = zip(*correct_preds_cosines)
        a_incorrect_cosines_mean, a_incorrect_cosines_std = zip(*incorrect_preds_cosines)

        ans_similarities[l]['correct_preds'] = {}
        ans_similarities[l]['correct_preds']['mean_cos_ha'] = np.mean(a_correct_cosines_mean)
        ans_similarities[l]['correct_preds']['std_cos_ha'] = np.std(a_correct_cosines_mean)
        ans_similarities[l]['correct_preds']['mean_std_cos_ha'] = np.mean(a_correct_cosines_std)
        ans_similarities[l]['correct_preds']['spearman_r'] = spearmanr(a_correct_cosines_mean, a_correct_cosines_std)
         
        ans_similarities[l]['incorrect_preds'] = {}
        ans_similarities[l]['incorrect_preds']['mean_cos_ha'] = np.mean(a_incorrect_cosines_mean)
        ans_similarities[l]['incorrect_preds']['std_cos_ha'] = np.std(a_incorrect_cosines_mean)
        ans_similarities[l]['incorrect_preds']['mean_std_cos_ha'] = np.mean(a_incorrect_cosines_std)
        ans_similarities[l]['incorrect_preds']['spearman_r'] = spearmanr(a_incorrect_cosines_mean, a_incorrect_cosines_std)

        #the following step might be necessary since number of incorrect model predicitions is significantly higher than the number of correct model predictions
        #draw different random samples from the set of cos(h_a) wrt incorrect answer predictions without (!) replacement 
        rnd_samples_incorrect_means = [np.random.choice(a_incorrect_cosines_mean, size=len(a_correct_cosines_mean), replace=False) for _ in range(5)]

        #compute independent t-tests and one-way ANOVAs
        ans_similarities[l]['ttest_p_val'] = np.mean([ttest_ind(a_correct_cosines_mean, rnd_sample)[1] for rnd_sample in rnd_samples_incorrect_means])
        ans_similarities[l]['anova_p_val'] = np.mean([f_oneway(a_correct_cosines_mean, rnd_sample)[1] for rnd_sample in rnd_samples_incorrect_means])

        """
        #plot CDFs w.r.t. cos(h_a) for both correct and erroneous model predictions across all transformer layers
        plot_cosine_cdfs(
                         a_correct_cosines_mean=np.asarray(a_correct_cosines_mean),
                         a_incorrect_cosines_mean=np.asarray(a_incorrect_cosines_mean),
                         source=source,
                         version=version,
                         layer_no=str(layer_no),
                         )

        #plot cos(h_a) distributions (i.e., PDFs) for both correct and erroneous model predictions across all transformer layers
        plot_cosine_distrib(
                            a_correct_cosines_mean=np.asarray(a_correct_cosines_mean),
                            a_incorrect_cosines_mean=np.asarray(a_incorrect_cosines_mean),
                            source=source,
                            version=version,
                            layer_no=str(layer_no),
                            )
        
        for boxplot_version in ['seaborn', 'matplotlib']:
            plot_cosine_boxplots(
                                 a_correct_cosines_mean=np.asarray(a_correct_cosines_mean),
                                 a_incorrect_cosines_mean=np.asarray(a_incorrect_cosines_mean),
                                 source=source,
                                 version=version,
                                 layer_no=str(layer_no),
                                 boxplot_version=boxplot_version,
                                )
        """

        if layer_no in est_layers:
            j += 1

    #TODO: the if statement below is just a work-around for now (must be fixed properly later)
    if not (version == 'test' and source.lower() == 'subjqa'):
        cos_similarities_preds = compute_rel_freq(cos_similarities_preds)

    ans_similarities = adjust_p_values(ans_similarities)

    if version == 'train':
        correct_preds_cosines_per_layer = np.asarray(correct_preds_cosines_per_layer)
        incorrect_preds_cosines_per_layer = np.asarray(incorrect_preds_cosines_per_layer)
        
        PATH = './results_hidden_reps/' + source.lower() + '/cosines'
        subdir_correct = '/correct'
        subdir_incorrect = '/incorrect'
        file_name = '/cosine_distrib' + '_' + layers + '.mat'
        
        if not os.path.exists(PATH + subdir_correct):
            os.makedirs(PATH + subdir_correct)
        if not os.path.exists(PATH + subdir_incorrect):
            os.makedirs(PATH + subdir_incorrect)

        io.savemat(PATH + subdir_correct + file_name,  mdict={'out': correct_preds_cosines_per_layer}, oned_as='row')
        io.savemat(PATH + subdir_incorrect + file_name,  mdict={'out': incorrect_preds_cosines_per_layer}, oned_as='row')

    return ans_similarities, cos_similarities_preds, X

def evaluate_estimations_and_cosines(
                                     test_results:dict,
                                     source:str,
                                     version:str,
                                     prediction:str,
                                     model_dir=None,
                                     n_epochs=None,
                                     batch_size=None,
                                     layers=None,
                                     w_strategy=None,
                                     computation=None,
                                     rnd_seed=None,
):
    pred_answers = test_results['predicted_answers']
    true_answers = test_results['true_answers']
    true_start_pos = np.asarray(test_results['true_start_pos'])
    true_end_pos = np.asarray(test_results['true_end_pos'])
    s_log_probs = np.asarray(test_results['start_log_probs'])
    e_log_probs = np.asarray(test_results['end_log_probs'])
    sent_pairs = test_results['sent_pairs']
    feat_reps = test_results['feat_reps']
    true_preds, pred_indices = [], []
    
    print("=============================================")
    print("===== Total number of predictions: {} =====".format(len(pred_answers)))
    print("=============================================")
    print()
    for i, pred_ans in enumerate(pred_answers):
        pred_ans = pred_ans.strip()
        true_ans = true_answers[i].strip()
        #NOTE: for now we exclusively want to estimate model predictions wrt answer spans that contain > 1 token
        if len(true_ans.split()) > 1:
            if compute_exact(true_ans, pred_ans):
                true_preds.append(1)
            else:
                true_preds.append(0)
            pred_indices.append(i)
    
    assert len(true_preds) == len(pred_indices)
    pred_indices = np.asarray(pred_indices)
    true_preds = np.asarray(true_preds)
    y = true_preds

    if prediction == 'learned':
        ans_similarities, cos_similarities_preds, X_cos = compute_similarities_across_layers(
                                                                                            feat_reps=feat_reps,
                                                                                            true_start_pos=true_start_pos,
                                                                                            true_end_pos=true_end_pos,
                                                                                            sent_pairs=sent_pairs,
                                                                                            pred_indices=pred_indices,
                                                                                            true_preds=true_preds,
                                                                                            s_log_probs=s_log_probs,
                                                                                            e_log_probs=e_log_probs,
                                                                                            source=source,
                                                                                            version=version,
                                                                                            layers=layers,
                                                                                            )

        if computation in ['concat', 'weighting']:
            #interpolate values wrt to *train* CDFs
            X_cos = interp_cos_per_layer(
                                         X=X_cos,
                                         source=source,
                                         version=version,
                                         layers=layers,
                                         w_strategy=w_strategy,
                                         computation=computation,
                                         y=y if version == 'train' or w_strategy == 'oracle' else None,
                                         concatenation='normal',
                                         )
            model_name = 'fc_nn' + '_' + w_strategy + '_' + computation + '_' + 'heuristic' + '_' + str(rnd_seed)

        #elif re.search(r'baseline', computation):
        #    X = compute_baseline_features(
        #                                  feat_reps=feat_reps,
        #                                  sent_pairs=sent_pairs,
        #                                  pred_indices=pred_indices,
        #                                  s_log_probs=s_log_probs,
        #                                  e_log_probs=e_log_probs,
        #                                  method='qa_concat' if re.search(r'concat', computation) else 'heuristic',
        #                                  )
        #    model_name = 'fc_nn' + '_' + computation + str(rnd_seed)
        else:
            model_name = 'fc_nn' + '_' + computation + '_' + 'heuristic' + '_'  + str(rnd_seed)

        X_baseline = compute_baseline_features(
                                              feat_reps=feat_reps,
                                              sent_pairs=sent_pairs,
                                              pred_indices=pred_indices,
                                              s_log_probs=s_log_probs,
                                              e_log_probs=e_log_probs,
                                              method='heuristic',
                                              )

        if re.search(r'baseline', computation):
            X = X_baseline
            model_name = 'fc_nn' + '_' + computation + str(rnd_seed)
        else:
            X = np.hstack((X_baseline, X_cos))

        M = X.shape[1] #M = number of input features (i.e., x $\in$ R^M)
        #X, y = shuffle_arrays(X, y) if version == 'train' else X, y #shuffle order of examples during training (this step is not necessary at inference time)
        tensor_ds = create_tensor_dataset(X, y)
        #dl = BatchGenerator(dataset=tensor_ds, batch_size=batch_size)
        dl = DataLoader(dataset=tensor_ds, batch_size=batch_size, shuffle=True if version =='train' else False)

        if version == 'train':
            """               
            clf = LogisticRegression(random_state=rnd_seed).fit(X, y)
            dump(clf, model_dir + '/' + model_name + '.joblib')
            y_hat = clf.predict(X)
            train_f1 = f1_score(y_true=y, y_pred=y_hat, average='macro')
            return ans_similarities, cos_similarities_preds, train_f1
            """
            y_distribution = Counter(y)
            y_weights = torch.tensor(y_distribution[0]/y_distribution[1], dtype=torch.float)
            model = FFNN(in_size=M)
            model.to(device)
            losses, f1_scores, model = train(model=model, train_dl=dl, version=version, n_epochs=n_epochs, batch_size=batch_size, y_weights=y_weights)
            torch.save(model.state_dict(), model_dir + '/%s' % (model_name)) #save model's weights
            return ans_similarities, cos_similarities_preds, losses, f1_scores, len(true_preds)

        else:
            """
            clf = load(model_dir + '/' + model_name + '.joblib') 
            y_hat = clf.predict(X)
            test_f1 = f1_score(y_true=y, y_pred=y_hat, average='macro')
            """
            model = FFNN(in_size=M)
            model.load_state_dict(torch.load(model_dir + '/%s' % (model_name))) #load model's weights
            model.to(device)
            test_f1, test_acc, incorrect_preds = test(model=model, test_dl=dl)
            
            #store Q&A, and corresponding answer span predictions for which the neural net failed to make a correct classification
            sent_pairs = np.asarray(sent_pairs)[pred_indices][incorrect_preds]
            true_answers = np.asarray(true_answers)[pred_indices][incorrect_preds]
            pred_answers = np.asarray(pred_answers)[pred_indices][incorrect_preds]
            qas = []
            for i, sent_pair in enumerate(sent_pairs):
                sent_pair = sent_pair.strip().split()
                q = sent_pair[:sent_pair.index('[SEP]')+1]
                a = true_answers[i].split()
                qas.append(' '.join(q + a + ['[SEP]']))
            qas = np.asarray(qas)
            X = X_cos[incorrect_preds][:, ::2]
            y = y[incorrect_preds]

            model_name = model_name.lstrip('fc_nn_')
            model_name =  "".join(filter(lambda char: not char.isdigit(), model_name)).rstrip('_')
            PATH = './incorrect_predictions/' + source.lower() + '/' + model_name + '/'

            if not os.path.exists(PATH):
                os.makedirs(PATH)

            with open(PATH + 'question_answers.txt', 'wb') as f:
                np.save(f, qas)

            with open(PATH + 'ans_span_preds.txt', 'wb') as f:
                np.save(f, pred_answers)

            with open(PATH + 'features.txt', 'wb') as f:
                np.save(f, X)

            with open(PATH + 'labels.txt', 'wb') as f:
                np.save(f, y)

            return ans_similarities, cos_similarities_preds, test_f1, test_acc, len(true_preds)
    else:
        y_distrib = Counter(y)
        if y_distrib[1] > y_distrib[0]:
            y_hat = np.ones(len(y))
        else:
            y_hat = np.zeros(len(y))
        acc = (y == y_hat).mean()    
        return f1_score(y_true=y, y_pred=y_hat, average='macro'), acc 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='SubjQA',
        help='Estimate model predictions (i.e., correct or erroneous) wrt hidden reps obtained from fine-tuning (and evaluating) on *source*.')
    parser.add_argument('--version', type=str, default='train',
        help='Must be one of {train, test}')
    parser.add_argument('--prediction', type=str, default='learned',
        help='If "learned", compute feature matrix X and labels vector y wrt cos(h_a) obtained from fine-tuning on *source* to train feed-forward neural net')
    parser.add_argument('--model_dir', type=str, default='./saved_models/ans_pred',
        help='Set model save directory for ans prediction model.')
    parser.add_argument('--batch_size', type=int, default=8,
        help='Specify mini-batch size.')
    parser.add_argument('--n_epochs', type=int, default=25,
        help='Set number of epochs model should be trained for.')
    parser.add_argument('--layers', type=str, default='',
        help='Must be one of {all_layers, top_three_layers}.')
    parser.add_argument('--w_strategy', type=str, default='',
        help='Must be one of {distance, cdf, oracle}.')
    
    args = parser.parse_args()
    #get hidden representations
    results, file_name = get_hidden_reps(source=args.source, version=args.version)
    #set NumPy random seed for reproducibility of results
    np.random.seed(42)
    
    if args.prediction == 'learned':

        #iterate over five different random seeds to obtain more robust results
        rnd_seeds = np.random.randint(0, 100, 5)
        computations = ['raw', 'concat', 'weighting', 'baseline_heuristic'] if args.w_strategy == 'distance' else ['concat', 'weighting']

        for computation in computations:
            hidden_reps_results = {}
            #perform computations for different random seeds (results might vary as a function of random seeds due to different initialisations)
            for k, rnd_seed in enumerate(rnd_seeds):
                np.random.seed(rnd_seed)
                torch.manual_seed(rnd_seed)

                try:
                    torch.cuda.manual_seed_all(rnd_seed)
                except:
                    pass
            
                if args.version == 'train':
                    assert isinstance(args.n_epochs, int), 'Number of epochs must be defined in train mode'
                    
                    if not os.path.exists(args.model_dir):
                        os.makedirs(args.model_dir)
                    
                    ans_similarities, cos_similarities_preds, losses, f1_scores, n_examples = evaluate_estimations_and_cosines(
                                                                                                                                test_results=results,
                                                                                                                                source=args.source,
                                                                                                                                prediction=args.prediction,
                                                                                                                                version=args.version,
                                                                                                                                model_dir=args.model_dir,
                                                                                                                                batch_size=args.batch_size,
                                                                                                                                n_epochs=args.n_epochs,
                                                                                                                                layers=args.layers,
                                                                                                                                w_strategy=args.w_strategy,
                                                                                                                                computation=computation,
                                                                                                                                rnd_seed=rnd_seed,
                                                                                                                                )
                    """
                    try:
                        hidden_reps_results['train_f1'] += train_f1
                    except KeyError:
                        hidden_reps_results['train_f1'] = train_f1

                    """
                    try:
                        hidden_reps_results['train_losses'].append(losses)
                        hidden_reps_results['train_f1s'].append(f1_scores)
                    except KeyError:
                        hidden_reps_results['train_losses'] = [losses]
                        hidden_reps_results['train_f1s'] = [f1_scores]

                else:
                    ans_similarities, cos_similarities_preds, test_f1, test_acc, n_examples = evaluate_estimations_and_cosines(
                                                                                                                             test_results=results,
                                                                                                                             source=args.source, 
                                                                                                                             prediction=args.prediction,
                                                                                                                             version=args.version,
                                                                                                                             model_dir=args.model_dir,
                                                                                                                             batch_size=args.batch_size,
                                                                                                                             layers=args.layers,
                                                                                                                             w_strategy=args.w_strategy,
                                                                                                                             computation=computation,
                                                                                                                             rnd_seed=rnd_seed,
                                                                                                                             )
                    try:
                        hidden_reps_results['test_f1'] += test_f1
                        hidden_reps_results['test_acc'] += test_acc
                    except KeyError:
                        hidden_reps_results['test_f1'] = test_f1
                        hidden_reps_results['test_acc'] = test_acc
            
                if k == 0:
                    hidden_reps_results['cos_similarities_true'] = ans_similarities
                    hidden_reps_results['cos_similarities_preds'] = cos_similarities_preds
                    hidden_reps_results['dataset_size'] = n_examples

            
            #compute mean F1 score
            #hidden_reps_results['train_f1' if args.version == 'train' else 'test_f1'] /= len(rnd_seeds)
            if args.version == 'test':
                hidden_reps_results['test_f1'] /= len(rnd_seeds)
                hidden_reps_results['test_acc'] /= len(rnd_seeds)

            #create PATH
            PATH = './results_hidden_reps/' + '/' + args.source.lower() + '/' + args.prediction + '/'
            if not os.path.exists(PATH):
                os.makedirs(PATH)

            #save results
            with open(PATH + file_name + '_' + args.layers + '_' + args.w_strategy + '_' + computation + '.json', 'w') as json_file:
                json.dump(hidden_reps_results, json_file)

    elif args.prediction == 'majority':
        hidden_reps_results = {}
        hidden_reps_results['f1_score'],  hidden_reps_results['acc']= evaluate_estimations_and_cosines(
                                                                                                       test_results=results,
                                                                                                       source=args.source,
                                                                                                       prediction=args.prediction, 
                                                                                                       version=args.version,
                                                                                                       )
        #create PATH
        PATH = './results_hidden_reps/' + '/' + args.source.lower() + '/' + args.prediction + '/'
        if not os.path.exists(PATH):
            os.makedirs(PATH)

        #save results
        with open(PATH + file_name + '_' + 'majority' + '.json', 'w') as json_file:
            json.dump(hidden_reps_results, json_file)

