__all__ = ['evaluate_estimations_and_cosines']

import argparse
import matplotlib
import json
import os
import re 

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from collections import defaultdict, Counter
from eval_squad import compute_exact
from scipy.stats import f_oneway, mode, ttest_ind
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

def compute_ans_similarities(a_hiddens:np.ndarray, metric:str):
    a_dists = []
    for i, a_i in enumerate(a_hiddens):
        for j, a_j in enumerate(a_hiddens):
            #NOTE: we don't want to compute cosine sim of a vector with itself (i.e., cos_sim(u, u) = 1)
            if i != j and j > i:
                a_dists.append(cosine_sim(u=a_i, v=a_j))
    return np.mean(a_dists), np.std(a_dists)


def plot_cosine_boxplots(
                         a_correct_cosines_mean:np.ndarray,
                         a_incorrect_cosines_mean:np.ndarray,
                         source:str,
                         dim:str,
                         layer_no:str,
                         boxplot_version:str,
):
    plt.figure(figsize=(6, 4), dpi=100)

    # set fontsize var
    lab_fontsize = 12

    ax = plt.subplot(111)

    # hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # only show ticks on the left (y-axis) and bottom (x-axis) spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    if boxplot_version == 'seaborn':
        
        sns.boxplot(
                     data=[a_correct_cosines_mean, a_incorrect_cosines_mean],
                     color='white',
                     meanline=False, # not necessary to show dotted mean line when showmeans = True (only set one of the two to True)
                     showmeans=True,
                     showcaps=True,
                     showfliers=True,
                     )

        ax.set_xticklabels(['correct', 'erroneous'])
        ax.set_xlabel('answer predictions', fontsize=lab_fontsize)
        ax.set_ylabel('cosine similarities', fontsize=lab_fontsize)

        for i, artist in enumerate(ax.artists):
            if i % 2 == 0:
                col = 'pink'
            else:
                col = 'steelblue'

            # this sets the color for the main box
            artist.set_edgecolor(col)
            
            # each box has 7 associated Line2D objects (to make the whiskers, median lines, means, fliers, etc.)
            # loop over them, and use the same colour as above (display means in black to make them more salient)
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
        plt.ylabel('cosine similarities', fontsize=lab_fontsize)

    plt.tight_layout()
    plt.savefig('./plots/hidden_reps/cosine_distributions/' + source.lower() + '/' + dim + '_' + 'dim' + '/' + 'boxplots' + '/' + 'layer' + '_' + layer_no +  '_' + boxplot_version + '.png')
    plt.clf()
    plt.close()

def plot_cosine_distrib(
                        a_correct_cosines_mean:np.ndarray,
                        a_incorrect_cosines_mean:np.ndarray,
                        source:str,
                        dim:str,
                        layer_no:str,
):
    # the higher the dpi, the better is the resolution of the plot (don't set dpi too high)
    plt.figure(figsize=(6, 4), dpi=100)

    # set vars
    legend_fontsize = 8
    lab_fontsize = 10

    ax = plt.subplot(111)

    # hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # only show ticks on the left (y-axis) and bottom (x-axis) spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
        
    sns.distplot(a_correct_cosines_mean, kde=True, norm_hist=True, label='correct answers')
    sns.distplot(a_incorrect_cosines_mean, kde=True, norm_hist=True, label='wrong answers')
    plt.xlabel('cosine similarities', fontsize=lab_fontsize)
    plt.ylabel('probability density', fontsize=lab_fontsize)
    plt.legend(fancybox=True, shadow=True, loc='best', fontsize=legend_fontsize)
    plt.tight_layout()
    plt.savefig('./plots/hidden_reps/cosine_distributions/' + source.lower() + '/' + dim + '_' + 'dim' + '/' + 'density_plots' + '/' + 'layer' + '_' + layer_no + '.png')
    plt.close()

def compute_similarities_across_layers(
                                       feat_reps:dict,
                                       true_start_pos:list,
                                       true_end_pos:list,
                                       sent_pairs:list,
                                       pred_indices:list,
                                       true_preds:np.ndarray,
                                       source:str,
                                       metric:str,
                                       dim:str,
                                       rnd_state:int=42,
):
    if dim == 'high':
        p_components = .95 #retain 90% or 95% of the hidden rep's variance (95% = top 57 principal components)
        cos_thresh = .45 if source.lower() == 'subjqa' else .50
        est_layers = [4, 5, 6] if source.lower() == 'subjqa' else [5, 6]
    else:
        p_components = 2 #keep top two or three principal components (analog to 2D / 3D plots)
        cos_thresh = .85 if source.lower() == 'subjqa' else .82
        std_thresh = .20
        est_layers = [4, 5, 6]

    # initialise PCA (we need to apply PCA to remove noise from the feature representations)
    pca = PCA(n_components=p_components, svd_solver='auto', random_state=rnd_state)

    ans_similarities = defaultdict(dict)
    N = len(pred_indices)    
    est_preds = []
    #est_preds = np.zeros(N)
    for l, hiddens_all_sents in feat_reps.items():
        layer_no = int(l.lstrip('Layer' + '_'))
        correct_preds_dists, incorrect_preds_dists = [], []
        k = 0
        if layer_no in est_layers:
           est_preds_current = np.zeros(N)

        for i, hiddens in enumerate(hiddens_all_sents):
            if i in pred_indices:
                hiddens = np.array(hiddens)
                sent = sent_pairs[i].strip().split()
                sep_idx = sent.index('[SEP]')

                # remove hidden reps for special [CLS] and [SEP] tokens
                #hiddens = np.vstack((hiddens[1:sep_idx, :], hiddens[sep_idx+1:-1, :])) 
                
                # transform feat reps into low-dim space with PCA
                hiddens = pca.fit_transform(hiddens)

                if layer_no == 1 and i == pred_indices[0]:
                    print("==============================================================")
                    print("=== Number of components in transformed hidden reps: {} ===".format(hiddens.shape[1]))
                    print("==============================================================")
                    print()

                # extract hidden reps for answer span
                #a_hiddens = hiddens[true_start_pos[i]-2:true_end_pos[i]-1, :] # move ans span indices two positions to the left
                a_hiddens = hiddens[true_start_pos[i]:true_end_pos[i]+1, :]

                # compute cosine similarities among hidden reps w.r.t. answer span
                a_mean_cos, a_std_cos = compute_ans_similarities(a_hiddens, metric)

                if true_preds[pred_indices == i] == 1:
                    correct_preds_dists.append((a_mean_cos, a_std_cos))
                else:
                    incorrect_preds_dists.append((a_mean_cos, a_std_cos))

                # estimate model predictions w.r.t. avg cosine similarities among answer hidden reps in the penultimate transformer layer
                if layer_no in est_layers:
                    if dim == 'high':
                        if a_mean_cos > cos_thresh: 
                            est_preds_current[k] += 1
                    else:
                        if layer_no == 6:
                            cos_thresh -= .20
                        if a_mean_cos > cos_thresh and a_std_cos < std_thresh:
                            est_preds_current[k] += 1
                k += 1

        # unzip means and stds w.r.t. cosine similarities
        a_correct_cosines_mean, a_correct_cosines_std = zip(*correct_preds_dists)
        a_incorrect_cosines_mean, a_incorrect_cosines_std = zip(*incorrect_preds_dists)

        ans_similarities[l]['correct_preds'] = {}
        ans_similarities[l]['correct_preds']['mean_cos_ha'] = np.mean(a_correct_cosines_mean)
        #ans_similarities[l]['correct_preds']['std_cos_ha'] = np.mean(a_correct_cosines_std)
        ans_similarities[l]['correct_preds']['std_cos_ha'] = np.std(a_correct_cosines_mean)
        
        ans_similarities[l]['incorrect_preds'] = {}
        ans_similarities[l]['incorrect_preds']['mean_cos_ha'] = np.mean(a_incorrect_cosines_mean)
        #ans_similarities[l]['incorrect_preds']['std_cos_ha'] = np.mean(a_incorrect_cosines_std)
        ans_similarities[l]['incorrect_preds']['std_cos_ha'] = np.std(a_incorrect_cosines_mean)

        #TODO: figure out whether equal_var should be set to False
        ans_similarities[l]['ttest_p_val'] = ttest_ind(a_correct_cosines_mean, a_incorrect_cosines_mean, equal_var=True)[1]
        ans_similarities[l]['anova_p_val'] = f_oneway(a_correct_cosines_mean, a_incorrect_cosines_mean)[1]


        # plot the cosine similarity distributions for both correct and incorrect model predictions across all transformer layers
        plot_cosine_distrib(
                            a_correct_cosines_mean=np.array(a_correct_cosines_mean),
                            a_incorrect_cosines_mean=np.array(a_incorrect_cosines_mean),
                            source=source,
                            dim=dim,
                            layer_no=str(layer_no),
                            )

        for boxplot_version in ['seaborn', 'matplotlib']:
            plot_cosine_boxplots(
                                a_correct_cosines_mean=np.array(a_correct_cosines_mean),
                                a_incorrect_cosines_mean=np.array(a_incorrect_cosines_mean),
                                source=source,
                                dim=dim,
                                layer_no=str(layer_no),
                                boxplot_version=boxplot_version,
                                )

        if layer_no in est_layers:
            est_preds.append(est_preds_current)

    est_preds = np.stack(est_preds, axis=1)
    #if all estimations w.r.t. both layer 4, 5 and 6 yield correct pred we assume a correct model pred else incorrect
    est_preds = np.array([1 if len(np.unique(row)) == 1 and np.unique(row)[0] == 1 else 0 for row in est_preds])
    return ans_similarities, est_preds

def evaluate_estimations_and_cosines(
                                     test_results:dict,
                                     source:str,
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
    ans_similarities, est_preds = compute_similarities_across_layers(
                                                                     feat_reps=feat_reps,
                                                                     true_start_pos=true_start_pos,
                                                                     true_end_pos=true_end_pos,
                                                                     sent_pairs=sent_pairs,
                                                                     pred_indices=pred_indices,
                                                                     true_preds=true_preds,
                                                                     source=source,
                                                                     metric=metric,
                                                                     dim=dim,
                                                                  )

    est_accs = {}
    est_accs['correct_preds'] = (true_preds[true_preds == 1] == est_preds[true_preds == 1]).mean() * 100
    est_accs['incorrect_preds'] = (true_preds[true_preds == 0] == est_preds[true_preds == 0]).mean() * 100
    est_accs['total_preds'] = {} 
    est_accs['total_preds']['acc'] = (true_preds == est_preds).mean() * 100
    est_accs['total_preds']['f1_weighted'] = f1_score(true_preds, est_preds, average='weighted') * 100
    #est_accs = {'Layer' + '_' + str(l + 1): (est_preds == true_preds).mean() * 100 for l, est_preds in enumerate(est_preds_top_layers)}
    return est_accs, ans_similarities

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='SubjQA',
        help='Estimate model predictions (i.e., correct or erroneous) w.r.t. hidden reps obtained from fine-tuning (and evaluating) on *source*.')
    args = parser.parse_args()
   
    # set variables
    source = args.source
    metric = 'cosine'
    dims = ['high', 'low']

    # get feat reps
    test_results, model_name = get_hidden_reps(source=source)

    # estimate model predictions w.r.t. answer and context hidden reps in latent space (per transformer layer) AND
    # compute (dis-)similarities among hidden representations in h_a for both correct and erroneous model predictions (at each layer)
    ests_and_cosines  = {dim: evaluate_estimations_and_cosines(test_results=test_results, source=source, metric=metric, dim=dim) for dim in dims}

    hidden_reps_results = {}
    hidden_reps_results['estimations'] = {dim: results[0] for dim, results in ests_and_cosines.items()}
    hidden_reps_results['cos_similarities'] = {dim: results[1] for dim, results in ests_and_cosines.items()}

    print()
    print(hidden_reps_results)
    print()

    # save results
    with open('./results_hidden_reps/' + '/' + source.lower() + '/' + model_name + '.json', 'w') as json_file:
        json.dump(hidden_reps_results, json_file)

