__all__ = [
           'get_results',
           'plot_results',
           'plotting',
]

import numpy as np
import pandas as pd
import matplotlib as mpl 
import matplotlib.pyplot as plt

import json
import os
import re

from collections import defaultdict
from itertools import islice

def get_results(
                task:str,
                version:str,
                model:str,
                task_setting:str,
                aux=None,
):
    subdir = './results_train/' if version == 'train' else './results_test/'
    subsubdir = subdir + task + '/' + model + '/' + task_setting + '/'
    subsubdir = subsubdir + aux + '/' if model == 'adversarial' else subsubdir
    all_files = list(map(lambda f: subsubdir + f, os.listdir(subsubdir)))
    all_results = defaultdict(dict)
    for file in all_files: #islice(all_files, 1, None):
        if not re.search(r'.ipynb_checkpoints', file):
            with open(file) as f:
                r = json.load(f)
                for metric, values in r.items():
                    all_results[file][metric] = values
    return dict(all_results)

def plot_results(
                 results:dict,
                 task:str,
                 model:str,
                 task_setting:str,
                 iv:str='datasets',
                 metric:str='',
                 aux=None,
                 correlation:bool=False,
):
    r_plot = defaultdict(dict) if correlation else {}
    for clf, r in results.items():
        for m, v in r.items():
            if correlation:
                if re.search(r'' + 'batch_f1', m):
                    r_plot[clf]['train'] = v
                elif re.search(r'' + 'val_f1', m):
                    r_plot[clf]['val'] = v
            else:
                if re.search(r'' + metric, m):
                    r_plot[clf] = v                   
                    
    if iv == 'datasets':
        params = [r'$D_{SQuAD} \cup D_{Subj}$', r'$D_{Subj}$']
        
    elif iv == 'models' and task_setting == 'single':
        params = ['Highway', 'Linear', 'BiLSTM + Highway', 'BiLSTM']
    
    elif iv == 'models' and task_setting == 'multi':
        params = [r'Adversarial Simple $(\mathbf{q, a})$', 
                  r'Normal $(\mathbf{q, a})$',
                  r'Adversarial GRL $(\mathbf{q, a})$ ', 
                  r'Adversarial Simple $(\mathbf{q, c})$',
                  r'Normal $(\mathbf{q, c})$',
                  r'GRL $(\mathbf{q, c})$',
                 ]            
        
    elif model == 'adversarial' and iv == 'methods':
        params = [r'$D_{SQuAD} \cup D_{Subj}$ $(\mathbf{q, c})$ Simple', 
                  r'$D_{Subj}$ $(\mathbf{q, c})$ Simple ', 
                  r'$D_{SQuAD} \cup D_{Subj}$ $(\mathbf{q, a})$ Simple ', 
                  r'$D_{Subj}$ $(\mathbf{q, a})$ Simple', 
                  r'$D_{SQuAD} \cup D_{Subj}$ $(\mathbf{q, a})$ GRL', 
                  r'$D_{Subj}$ $(\mathbf{q, a})$ GRL', 
                  r'$D_{SQuAD} \cup D_{Subj}$ $(\mathbf{q, c})$ GRL',
                  r'$D_{Subj}$ $(\mathbf{q, c})$ GRL',
                 ]
    
    if re.search(r'batch_loss', metric):
        plt.figure(figsize=(10, 6), dpi=100)
    else:
        plt.figure(figsize=(8, 4), dpi=90)
    
    # set font sizes
    y_lab_fontsize = 12
    x_lab_fontsize = y_lab_fontsize
    title_fontsize = 13
    legend_fontsize = 8
    
    ax = plt.subplot(111)
        
    for idx, (clf, met) in enumerate(r_plot.items()):
        #print(clf)
        if correlation:
            try:
                ax.plot(met['train'], met['val'], '-o', label=params[idx])
            except ValueError:
                ax.plot(met['train'][:-1], met['val'], '-o', label=params[idx])
        else:
            ax.plot(range(1, len(met) + 1), met, '-o', label=params[idx])
    
    # hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # only show ticks on the left (y-axis) and bottom (x-axis) spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
    if correlation:
        ax.set_ylabel('Val F1', fontsize=y_lab_fontsize, labelpad=None)
            
    elif re.search(r'acc', metric): 
        ax.set_ylabel('Exact-match', fontsize=y_lab_fontsize, labelpad=None)
    
    elif re.search(r'f1', metric):
        ax.set_ylabel('F1', fontsize=y_lab_fontsize, labelpad=None)
    
    elif re.search(r'loss', metric):
        if metric == 'batch_loss':
            ax.set_yticks(range(1, 7))
        ax.set_ylabel('Loss', fontsize=y_lab_fontsize, labelpad=None) # TODO: adjust y-label position accordingly
    
    ax.set_xlabel('Train F1' if correlation else 'Evaluation steps', fontsize=x_lab_fontsize)
    
    if model == 'adversarial' and re.search(r'loss', metric):
        ax.legend(fancybox=True,
                  shadow=True,
                  loc='upper right',
                  fontsize=legend_fontsize)
        
    elif model == 'adversarial' and not re.search(r'loss', metric):
        ax.legend(fancybox=True,
                  shadow=True,
                  bbox_to_anchor=(1.02, 0.5),
                  ncol=1,
                  fontsize=legend_fontsize)
    else:
        ax.legend(fancybox=True,
                  shadow=True,
                  loc='lower right' if re.search(r'acc', metric) or re.search(r'f1', metric) else 'best',
                  fontsize=legend_fontsize)
    
    if model == 'SubjQA':
        ax.set_title(r'$D_{SubjQA}$', fontsize=title_fontsize)
    elif model == 'combined':
        ax.set_title(r'$D_{SubjQA} \cup D_{SQuAD}$', fontsize=title_fontsize)
    else:
        ax.set_title('MTL' + ' ' + model.capitalize() if model == 'adversarial' else re.sub(r'_', ' ', model).strip().capitalize(),
                     fontsize=title_fontsize)
    
    plt.tight_layout()
    
    if model == 'adversarial':
        plt.savefig('./plots/' + 'models/' + task + '/' + model + '/' + task_setting + '/' + aux + '/' + metric + '.png')
    elif model == 'adversarial' and correlation:
        plt.savefig('./plots/' + 'models/' + task + '/' + model + '/' + task_setting + '/'  + aux + '/' +  'train_vs_val' + '.png')
    elif correlation:
        plt.savefig('./plots/' + 'models/' + task + '/' + model + '/' + task_setting + '/'  +  'train_vs_val' + '.png')
    else:
        plt.savefig('./plots/' + 'models/' + task + '/' + model + '/' + task_setting + '/' + metric + '.png')
        
    plt.show()
    plt.clf()
    
def plotting(
             models:list,
             metrics:list,
             task:str,
             version:str,
             task_setting:str,
             iv:str,
             aux=None,
):
    print('===========================')
    print('====== Task: {} ======'.format(task.upper()))
    print('===========================')
    print()
    for i, model in enumerate(models):
        # load results
        if model == 'adversarial':
            all_results = get_results(task=task, version=version, model=model, task_setting='multi', aux=aux)
        else:
            all_results = get_results(task=task, version=version, model=model, task_setting=task_setting, aux=None)
        print('==============================')
        print('====== Model: {} ======'.format(model.upper()))
        print('==============================')
        print()
        for j, metric in enumerate(metrics):
            # plot results
            print('=============================')
            print('===== Metric: {} ====='.format(metric.upper()))
            print('=============================')
            if model == 'adversarial':
                plot_results(all_results, task=task, metric=metric, iv='methods',  model=model, task_setting='multi', aux=aux)
            else:
                plot_results(all_results, task=task, metric=metric, iv=iv,  model=model, task_setting=task_setting, aux=None)