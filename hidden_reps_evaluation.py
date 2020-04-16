__all__ = ['evaluate_estimations']

import json
import os 

import numpy as np

from collections import defaultdict
from eval_squad import compute_exact


def euclidean_dist(u:np.ndarray, v:np.ndarray): return np.linalg.norm(u-v)

def cosine_sim(x:np.ndarray, y:np.ndarray):
    num = x @ y
    denom = np.linalg.norm(x) * np.linalg.norm(y) # default is Frobenius norm (i.e., L2 norm)
    return num / denom

def compute_ans_distances(a_hiddens:np.ndarray, metric:str):
    a_mean_dist = 0
    count = 0
    for i, a_i in enumerate(a_hiddens):
        for j, a_j in enumerate(a_hiddens):
            #NOTE: we don't want to compute cosine sim of a vector with itself (cos_sim = 1),
            #AND it's redundant to compute cosine sim twice for some pair of vectors
            if i != j and j > i:
                if metric == 'cosine':
                    a_mean_dist += cosine_sim(a_i, a_j)
            elif metric == 'euclid':
                    a_mean_dist += euclidean_dist(a_i, a_j)
            count += 1
    a_mean_dist /= count
    return a_mean_dist

def evaluate_estimations(test_results:dict, metric:str='cosine'):
    
    pred_answers = test_results['predicted_answers']
    true_answers = test_results['true_answers']
    true_start_pos = test_results['true_start_pos']
    true_end_pos = test_results['true_end_pos']
    sent_pairs = test_results['sent_pairs']
    feat_reps = test_results['feat_reps']
    true_preds, ans_to_pred_indices = [], []
    
    for i, pred_ans in enumerate(pred_answers):
        #NOTE: we exclusively want to make predictions for answer spans that contain more than 1 token
        if len(true_answers[i].strip().split()) > 1:
            if compute_exact(true_answers[i], pred_ans):
                true_preds.append(1)
            else:
                true_preds.append(0)
            ans_to_pred_indices.append(i)
            
    true_preds = np.array(true_predictions)
    est_preds = estimate_preds_based_on_distances(feat_reps, metric, true_start_pos, true_end_pos, sent_pairs, ans_to_pred_indices)
    est_accs = {'Layer' + '_' + str(l): (est_pred == true_preds).mean() * 100 for l, est_pred in enumerate(est_preds)}
    return est_accs

def estimate_preds_based_on_distances(
                                      feat_reps_per_layer:dict,
                                      metric:str,
                                      true_start_pos:list,
                                      true_end_pos:list,
                                      sent_pairs:list,
                                      ans_to_pred_indices:list,
):
    preds_top_three_layers = []
    for l, hiddens_all_sent_pairs in feat_reps_per_layer.items():
        N = len(ans_to_pred_indices)
        preds_current_layer = np.zeros(N)
        for i, hiddens in enumerate(hiddens_all_sent_pairs):
            if i in ans_indices:
                if int(l.lstrip('Layer_')) > 3:
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
                            preds_current_layer[i] += 1

                    elif metric == 'euclid':
                        euclid_dists_a_and_c = np.array([euclidean_dist(c_hidden, a_mean_rep) for c_hidden in c_hiddens])
                        c_closest_idx = np.argmin(cos_sims_a_and_c)
                        c_closest = c_hiddens[c_closest_idx]
                        a_mean_dist = compute_ans_distances(a_hiddens, metric)

                        if a_mean_dist < euclid_dists_a_and_c[c_closest_idx]:
                            preds_current_layer[i] += 1
        preds_top_three_layers.append(preds_current_layer)
    return preds_top_three_layers    