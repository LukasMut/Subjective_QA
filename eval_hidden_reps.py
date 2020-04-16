import argparse
import json
import os
import random
import re 

import numpy as np

from collections import defaultdict
from eval_squad import compute_exact
from sklearn.decomposition import PCA


def get_hidden_reps(source:str='SubjQA'):
    # set folder and subdirectories
    folder = '/results_test/'
    subdir = '/feat_reps/'
    subsubdir = '/qa_per_token/'
    task = subsubdir.lstrip('/').rstrip('/').lower()

    # create PATH
    cwd = '.'
    PATH = cwd + folder + subdir + subsubdir
    PATH += '/bert_stl_finetuned_subjqa/' if source == 'SubjQA' else '/bert_stl_finetuned_squad/'

    # we want to exclusively capture .json files
    files = [file for file in os.listdir(PATH) if file.endswith('.json')]
    f = files.pop()

    # load file
    with open(PATH + f) as json_file:
        test_results = json.load(json_file)
        model_name = 'hidden_rep_distances' + '_' + task
        print("===============================================================")
        print("======= File loaded: {} =======".format(model_name))
        print("===============================================================")
        print()

    return test_results, model_name

def euclidean_dist(u:np.ndarray, v:np.ndarray): return np.linalg.norm(u-v) # default is L2 norm

def cosine_sim(x:np.ndarray, y:np.ndarray):
    num = x @ y
    denom = np.linalg.norm(x) * np.linalg.norm(y) # default is Frobenius norm (i.e., L2 norm)
    return num / denom

def compute_ans_distances(a_hiddens:np.ndarray, metric:str):
    a_mean_dist = 0
    count = 0
    for i, a_i in enumerate(a_hiddens):
        for j, a_j in enumerate(a_hiddens):
            #NOTE: we don't want to compute cosine sim of a vector with itself (i.e., cos_sim = 1),
            #AND it's redundant to compute cosine sim (or Euclid dist) twice for some pair of vectors (i.e., cos_sim(u, v) == cos_sim(v, u))
            if i != j and j > i:
                if metric == 'cosine':
                    a_mean_dist += cosine_sim(a_i, a_j)
                elif metric == 'euclid':
                    a_mean_dist += euclidean_dist(a_i, a_j)
                count += 1
    a_mean_dist /= count
    return a_mean_dist

def evaluate_estimations(
                         test_results:dict,
                         metric:str='cosine',
                         dimensionality:str='high',
):
    pred_answers = test_results['predicted_answers']
    true_answers = test_results['true_answers']
    true_start_pos = test_results['true_start_pos']
    true_end_pos = test_results['true_end_pos']
    sent_pairs = test_results['sent_pairs']
    feat_reps = test_results['feat_reps']
    true_preds, pred_indices = [], []
    
    for i, pred_ans in enumerate(pred_answers):
        #NOTE: we exclusively want to make predictions for answer spans that contain more than 1 token
        if len(true_answers[i].strip().split()) > 1:
            if compute_exact(true_answers[i], pred_ans):
                true_preds.append(1)
            else:
                true_preds.append(0)
            pred_indices.append(i)
            
    true_preds = np.array(true_preds)
    print()
    print(len(true_preds))
    print()
    # estimate model predictions w.r.t. hidden reps
    est_preds = estimate_preds_wrt_hiddens(feat_reps, true_start_pos, true_end_pos, sent_pairs, pred_indices, metric, dimensionality)
    est_accs = {'Layer' + '_' + str(l): (est_pred == true_preds).mean() * 100 for l, est_pred in enumerate(est_preds)}
    return est_accs

def estimate_preds_wrt_hiddens(
                               feat_reps:dict,
                               true_start_pos:list,
                               true_end_pos:list,
                               sent_pairs:list,
                               pred_indices:list,
                               metric:str,
                               dimensionality:str,
                               rnd_state:int=42,
):  
    if dimensionality == 'low':
        #assert metric == 'euclid', 'Computing the cosine similarity between (word) vectors in low-dimensional (vector) space is not particularly useful. Thus, we must calculate the euclidean distance instead.'
        pca = PCA(n_components=.95, svd_solver='full', random_state=rnd_state) # init PCA

    est_preds_top_layers = []
    for l, hiddens_all_sent_pairs in feat_reps.items():
        #if int(l.lstrip('Layer_')) > 3:
        N = len(pred_indices)
        est_preds_current_layer = np.zeros(N)
        k = 0 # k = idx for ans spans for which we want to estimate preds based on hidden reps

        for i, hiddens in enumerate(hiddens_all_sent_pairs):
            if i in pred_indices:
                hiddens = np.array(hiddens)

                if dimensionality == 'low':
                    # transform feat reps into low-dim space with PCA
                    hiddens = pca.fit_transform(hiddens)

                sent = sent_pairs[i].split()
                sep_idx = sent.index('[SEP]')
                a_indices = np.arange(true_start_pos[i], true_end_pos[i]+1)
                a_hiddens = hiddens[a_indices, :]
                c_hiddens = np.vstack((hiddens[sep_idx+1:a_indices[0], :], hiddens[a_indices[-1]+1:-1, :]))
                a_mean_rep = np.mean(a_hiddens, axis=0)

                if metric == 'cosine':
                    cos_sims_a_and_c = np.array([cosine_sim(c_hidden, a_mean_rep) for c_hidden in c_hiddens])
                    c_most_sim_idx = np.argmax(cos_sims_a_and_c)
                    c_most_sim = c_hiddens[c_most_sim_idx]
                    a_mean_cos = compute_ans_distances(a_hiddens, metric)

                    if a_mean_cos > cos_sims_a_and_c[c_most_sim_idx]:
                        est_preds_current_layer[k] += 1

                elif metric == 'euclid':
                    euclid_dists_a_and_c = np.array([euclidean_dist(c_hidden, a_mean_rep) for c_hidden in c_hiddens])
                    c_closest_idx = np.argmin(cos_sims_a_and_c)
                    c_closest = c_hiddens[c_closest_idx]
                    a_mean_dist = compute_ans_distances(a_hiddens, metric)

                    if a_mean_dist < euclid_dists_a_and_c[c_closest_idx]:
                        est_preds_current_layer[k] += 1

                k += 1

        est_preds_top_layers.append(est_preds_current_layer)
    return est_preds_top_layers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='SubjQA',
        help='Make estimations based on hidden reps obtained from fine-tuning (and evaluating) on *source*.')
    args = parser.parse_args()
    # define variables
    source = args.source
    metrics = ['cosine', 'euclid']
    dims = ['high', 'low']
    # get feat reps
    test_results, model_name = get_hidden_reps(source=source)
    # estimate model predictions w.r.t. hidden reps in latent space (per transformer layer)
    est_per_metric = {metric: {dim: evaluate_estimations(test_results, metric, dim) for dim in dims} for metric in metrics}
    print()
    print(est_per_metric)
    print()
    # save results
    with open('./results_hidden_reps/' + '/' + source.lower() + '/' + model_name + '.json', 'w') as json_file:
        json.dump(est_per_metric, json_file)
