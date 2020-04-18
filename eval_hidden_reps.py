__all__ = ['evaluate_estimations']

import argparse
import json
import os
import random
import re 

import numpy as np

from collections import defaultdict, Counter
from eval_squad import compute_exact
from scipy.spatial.distance import mahalanobis
from scipy.stats import mode 
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score


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

def cosine_sim(u:np.ndarray, v:np.ndarray):
    num = u @ v
    denom = np.linalg.norm(u) * np.linalg.norm(v) # default is Frobenius norm (i.e., L2 norm)
    return num / denom

def compute_ans_distances(a_hiddens:np.ndarray, metric:str):
    a_dists = []
    for i, a_i in enumerate(a_hiddens):
        for j, a_j in enumerate(a_hiddens):
            #NOTE: we don't want to compute cosine sim of a vector with itself (i.e., cos_sim = 1)
            if i != j and j > i:
                a_dists.append(cosine_sim(u=a_i, v=a_j))
    return np.mean(a_dists), np.std(a_dists)

def evaluate_estimations(
                         test_results:dict,
                         metric:str,
                         dim:str,
):
    pred_answers = test_results['predicted_answers']
    true_answers = test_results['true_answers']
    true_start_pos = test_results['true_start_pos']
    true_end_pos = test_results['true_end_pos']
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
        #NOTE: for now we exclusively want to estimate model predictions w.r.t. answer spans that contain > 1 token
        if len(true_ans.split()) > 1:
            if compute_exact(true_ans, pred_ans):
                true_preds.append(1)
                #pred_indices.append(i)
            else:
                #pass
                true_preds.append(0)
            pred_indices.append(i)
    
    assert len(true_preds) == len(pred_indices)
    pred_indices = np.array(pred_indices)
    true_preds = np.array(true_preds)

    print()
    print(Counter(true_preds))
    print()

    # compute (dis-)similarities among hidden reps in H_a for both correct and erroneous model predictions at each layer
    # AND estimate model predictions w.r.t. hidden reps in the penultimate layer
    ans_similarities, est_preds = compute_distances_across_layers(
                                                                   feat_reps=feat_reps,
                                                                   true_start_pos=true_start_pos,
                                                                   true_end_pos=true_end_pos,
                                                                   sent_pairs=sent_pairs,
                                                                   pred_indices=pred_indices,
                                                                   true_preds=true_preds,
                                                                   metric=metric,
                                                                   dim=dim,
                                                                   )

    """
    
    est_preds_top_layers = estimate_preds_wrt_hiddens(
                                                      feat_reps=feat_reps,
                                                      true_start_pos=true_start_pos,
                                                      true_end_pos=true_end_pos,
                                                      sent_pairs=sent_pairs,
                                                      pred_indices=pred_indices,
                                                      metric=metric,
                                                      dim=dim,
                                                      )
    """
    est_accs = {}
    est_accs['correct_preds'] = (true_preds[true_preds == 1] == est_preds[true_preds == 1]).mean() * 100
    est_accs['incorrect_preds'] = (true_preds[true_preds == 0] == est_preds[true_preds == 0]).mean() * 100
    est_accs['total_preds'] = {} 
    est_accs['total_preds']['acc'] = (true_preds == est_preds).mean() * 100
    est_accs['total_preds']['f1_macro'] = f1_score(true_preds, est_preds, average='macro') * 100
    #est_accs = {'Layer' + '_' + str(l + 1): (est_preds == true_preds).mean() * 100 for l, est_preds in enumerate(est_preds_top_layers)}
    return est_accs, ans_similarities

def compute_distances_across_layers(
                                    feat_reps:dict,
                                    true_start_pos:list,
                                    true_end_pos:list,
                                    sent_pairs:list,
                                    pred_indices:list,
                                    true_preds:np.ndarray,
                                    metric:str,
                                    dim:str,
                                    p_components:int=2,
                                    rnd_state:int=42,
                                    est_layer:int=5,
                                    cos_thresh_layer_4:int=.90,
                                    std_thresh_layer_4:int=.10,
                                    cos_thresh_layer_5:int=.89,
                                    std_thresh_layer_5:int=.15,
):
    if dim == 'low':
        pca = PCA(n_components=p_components, svd_solver='auto', random_state=rnd_state)

    ans_similarities = defaultdict(dict)
    N = len(pred_indices)
    #est_preds = []
    est_preds = np.zeros(N)

    for l, hiddens_all_sents in feat_reps.items():
        layer_no = int(l.lstrip('Layer' + '_'))
        correct_preds_dists, incorrect_preds_dists = [], []
        k = 0
        #if layer_no in est_layers:
        #   est_preds_current = np.zeros(N)
        for i, hiddens in enumerate(hiddens_all_sents):
            if i in pred_indices:
                sent = sent_pairs[i].split()
                sep_idx = sent.index('[SEP]')
                hiddens = np.array(hiddens)
                
                if dim == 'low':
                    # transform feat reps into low-dim space with PCA
                    hiddens = pca.fit_transform(hiddens)

                # extract hidden reps for both answer and context 
                a_hiddens = hiddens[true_start_pos[i]:true_end_pos[i]+1, :]
                c_hiddens = np.vstack((hiddens[sep_idx+1:true_start_pos[i]], hiddens[true_end_pos[i]+1:-1,:]))

                # compute cosine similarities among hidden reps in w.r.t. answer span
                a_mean_dist, a_std_dist = compute_ans_distances(a_hiddens, metric)

                if true_preds[pred_indices == i] == 1:
                    correct_preds_dists.append((a_mean_dist, a_std_dist))
                else:
                    incorrect_preds_dists.append((a_mean_dist, a_std_dist))

                # estimate model predictions w.r.t. avg cosine similarities among answer hidden reps in the penultimate transformer layer
                if layer_no == est_layer:
                    if a_mean_dist > cos_thresh and a_std_dist < std_thresh:
                        est_preds[k] += 1
                k += 1

        ans_similarities[l]['correct_preds'] = {}
        ans_similarities[l]['correct_preds']['mean_cos_ha'] = np.mean(list(map(lambda d: d[0], correct_preds_dists)))
        ans_similarities[l]['correct_preds']['std_cos_ha'] = np.mean(list(map(lambda d: d[1], correct_preds_dists)))

        ans_similarities[l]['incorrect_preds'] = {}
        ans_similarities[l]['incorrect_preds']['mean_cos_ha'] = np.mean(list(map(lambda d: d[0], incorrect_preds_dists)))
        ans_similarities[l]['incorrect_preds']['std_cos_ha'] = np.std(list(map(lambda d: d[1], incorrect_preds_dists)))

        #if layer_no in est_layers:
        #   est_preds.append(est_preds_current)
    #est_preds = np.stack(est_preds, axis=1)

    #if all estimations w.r.t. both layer 4 and 5 yield correct pred we assume a correct model pred else incorrect
    #est_preds = np.array([1 if len(np.unique(row)) == 1 and np.unique(row)[0] == 1 else 0 for row in est_preds])
    return ans_similarities, est_preds

def estimate_preds_wrt_hiddens(
                               feat_reps:dict,
                               true_start_pos:list,
                               true_end_pos:list,
                               sent_pairs:list,
                               pred_indices:list,
                               metric:str,
                               dim:str,
                               rnd_state:int=42,
):  
    if dim == 'low':
        pca = PCA(n_components=2, svd_solver='auto', random_state=rnd_state) # initialise PCA

    est_preds_top_layers = []
    for l, hiddens_all_sent_pairs in feat_reps.items():
        layer_no = int(l.lstrip('Layer' + '_'))
        # estimate model predictions based on context and answer clusters w.r.t. hidden reps in top 3 layers
        if layer_no >= 4:
            N = len(pred_indices)
            est_preds_current_layer = np.zeros(N)
            k = 0
            for i, hiddens in enumerate(hiddens_all_sent_pairs):
                if i in pred_indices:
                    sent = sent_pairs[i].split()
                    sep_idx = sent.index('[SEP]')
                    hiddens = np.array(hiddens)
                    # transform feat reps into low-dim space with PCA (only useful for computing Euclidean distances; not necessary for cosine or Mahalanobis)
                    if dim == 'low':
                        hiddens = pca.fit_transform(hiddens)

                    a_indices = np.arange(true_start_pos[i], true_end_pos[i]+1)
                    q_hiddens = hiddens[1:sep_idx, :]
                    c_hiddens = np.vstack((hiddens[sep_idx+1:a_indices[0], :], hiddens[a_indices[-1]+1:-1, :]))
                    a_hiddens = hiddens[a_indices, :]    
                    a_mean_rep = a_hiddens.mean(0)

                    if metric == 'cosine':

                        cos_sims_a_and_c = np.array([cosine_sim(u=a_mean_rep, v=c_hidden) for c_hidden in c_hiddens])
                        c_most_sim_idx = np.argmax(cos_sims_a_and_c) # similarity measure ==> argmax
                        c_most_sim = c_hiddens[c_most_sim_idx]
                        c_most_sim_mean_cos = np.mean([cosine_sim(u=c_most_sim, v=a_hidden) for a_hidden in a_hiddens])
                        a_mean_cos = compute_ans_distances(a_hiddens, metric)

                        if a_mean_cos > c_most_sim_mean_cos: 
                            est_preds_current_layer[k] += 1

                    elif metric == 'euclid':

                        euclid_dists_a_and_c = np.array([euclidean_dist(u=a_mean_rep, v=c_hidden) for c_hidden in c_hiddens])
                        c_closest_idx = np.argmin(euclid_dists_a_and_c) # dissimilarity measure ==> argmin
                        c_closest = c_hiddens[c_closest_idx]
                        c_closest_mean_dist = np.mean([euclidean_dist(u=c_closest, v=a_hidden) for a_hidden in a_hiddens])
                        a_mean_dist = compute_ans_distances(a_hiddens, metric)

                        if a_mean_dist < c_closest_mean_dist:
                            est_preds_current_layer[k] += 1
                    k += 1
            est_preds_top_layers.append(est_preds_current_layer)
    est_preds_top_layers = np.stack(est_preds_top_layers, axis=1)

    #############################################################################################################################################
    #### ALTERNATIVE 1: IF MODE ACROSS ESTIMATIONS W.R.T. TOP THREE LAYERS YIELDS CORRECT PRED WE ASSUME A CORRECT MODEL PRED ELSE INCORRECT ####
    ######## ALTERNATIVE 2: IF ALL ESTIMATIONS W.R.T. TOP THREE LAYERS YIELD CORRECT PRED WE ASSUME A CORRECT MODEL PRED ELSE INCORRECT #########
    #############################################################################################################################################

    #est_preds_top_layers = mode(est_preds_top_layers, axis=1).mode.reshape(-1)
    est_preds_top_layers = np.array([1 if len(np.unique(row)) == 1 and np.unique(row)[0] == 1 else 0 for row in est_preds_top_layers])
    return est_preds_top_layers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='SubjQA',
        help='Estimate model predictions (i.e., correct or erroneous) w.r.t. hidden reps obtained from fine-tuning (and evaluating) on *source*.')
    args = parser.parse_args()
    # define variables
    source = args.source
    metrics = ['cosine']#, 'euclid']
    dims = ['low'] #['high', 'low']
    # get feat reps
    test_results, model_name = get_hidden_reps(source=source)
    # estimate model predictions w.r.t. answer and context hidden reps in latent space (per transformer layer) AND
    # compute (dis-)similarities among hidden representations in h_a for both correct and erroneous model predictions (at each layer)
    ests_and_dists  = {metric: {dim: evaluate_estimations(test_results, metric, dim) for dim in dims} for metric in metrics}

    hidden_reps_results = {}
    hidden_reps_results['estimations'] = {metric: {dim: results[0] for dim, results in dims.items()} for metric, dims in ests_and_dists.items()}
    hidden_reps_results['similarities'] = {metric: {dim: results[1] for dim, results in dims.items()} for metric, dims in ests_and_dists.items()}

    print()
    print(hidden_reps_results)
    print()

    # save results
    with open('./results_hidden_reps/' + '/' + source.lower() + '/' + model_name + '.json', 'w') as json_file:
        json.dump(hidden_reps_results, json_file)

