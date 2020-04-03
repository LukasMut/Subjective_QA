__all__ = [
           'conf_mat',
           'get_results',
           'plot_results',
           'plotting',
           'plot_confusion_matrix',
           'plot_seqs_projected_via_tsne',
           'plot_feat_reps_per_layer',
]

import numpy as np
import pandas as pd
import matplotlib as mpl 
import matplotlib.pyplot as plt

import json
import os
import re

from collections import defaultdict
from itertools import islice, product
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.utils import check_matplotlib_support
from sklearn.utils.multiclass import unique_labels


def get_results(
                task:str,
                version:str,
                model:str,
                task_setting:str,
                layer=None,
                aux=None,
                aux_task=None,
                task_sampling=None,
):
    subdir = './results_train/' if version == 'train' else './results_test/'
    subsubdir = subdir + task + '/' + model + '/' + task_setting + '/'
    
    if model in ['SubjQA', 'combined'] and task_setting == 'multi':
        assert isinstance(layer, str), 'When comparing across datasets in MTL setting, subfolder for model type must be provided'
        subsubdir += layer + '/'
        assert isinstance(aux, str), 'When comparing across datasets in MTL setting, subfolder for number of aux tasks must be provided'
        subsubdir += aux + '/'
        assert isinstance(task_sampling, str), 'When comparing across datasets in MTL setting, sampling strategy must be provided'
        subsubdir += task_sampling + '/'
        
    elif model == 'adversarial':
        subsubdir += aux + '/'
        if aux == 'aux_1':
            assert isinstance(aux_task, str), 'When comparing across adversarial models in aux_1, aux task must be defined'
            subsubdir += aux_task + '/'
            assert isinstance(layer, str), 'When comparing across adversarial models in aux_1, subfolder for model type must be provided'
            subsubdir += layer + '/'
            assert isinstance(task_sampling, str), 'When comparing across adversarial models in aux_1 sampling strategy must be provided'
            subsubdir += task_sampling + '/'
            
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
                 layer=None,
                 aux=None,
                 aux_task=None,
                 task_sampling=None,
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
                    
    if task =='sbj_class' and iv == 'datasets' and task_setting == 'single':
        params = [
                  r'$D_{SQuAD} \cup D_{Subj}$ $(\mathbf{q, a})$',
                  r'$D_{Subj}$ $(\mathbf{q, a})$',
                  r'$D_{SQuAD} \cup D_{Subj}$ $(\mathbf{q, c})$',
                  r'$D_{Subj}$ $(\mathbf{q, c})$',
        ]
        
    elif task =='domain_class' and iv == 'datasets' and task_setting == 'single':
        params = [
                  r'$D_{SQuAD} \cup D_{Subj}$',
                  r'$D_{Subj}$',
        ]
  
    elif task == 'QA' and iv == 'datasets' and task_setting == 'single':
        params = [
                  r'$D_{SQuAD} \cup D_{Subj}$',
                  r'$D_{Subj}$',
        ]
        
    elif task =='domain_class' and iv == 'models' and task_setting == 'single':
        params = [
                  r'Highway',
                  r'Linear',
                  r'BiLSTM',
        ]
        
    elif task =='sbj_class' and iv == 'models' and task_setting == 'single':
        params = [
                  r'Linear $(\mathbf{q, a})$',
                  r'Linear $(\mathbf{q, c})$',
                  r'Highway $(\mathbf{q, a})$',
                  r'Highway $(\mathbf{q, c})$',
        ]

    elif iv == 'models' and task_setting == 'single':
        params = [
                  r'Highway',
                  r'Linear',
                  r'BiLSTM',
        ]
    
    elif iv == 'models' and task_setting == 'multi':
        params = [
                  r'Adversarial Simple $(\mathbf{q, a})$', 
                  r'Normal $(\mathbf{q, a})$',
                  r'Adversarial GRL $(\mathbf{q, a})$ ', 
                  r'Adversarial Simple $(\mathbf{q, c})$',
                  r'Normal $(\mathbf{q, c})$',
                  r'GRL $(\mathbf{q, c})$',
                 ]
        
    elif (model == 'adversarial' or model == 'linear') and aux == 'aux_1' and aux_task == 'domain_class':
        params = [
          r'$D_{Subj}$ $(\mathbf{q, c})$ Simple ', 
          r'$D_{Subj}$ $(\mathbf{q, c})$ GRL',
         ]
    
    elif model == 'adversarial' and iv == 'methods':
        params = [
                  r'$D_{SQuAD} \cup D_{Subj}$ $(\mathbf{q, c})$ Simple', 
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
                ax.plot(met['train'], met['val'], '-^', label=params[idx])
            except ValueError:
                ax.plot(met['train'][:-1], met['val'], '-^', label=params[idx])
        else:
            ax.plot(range(1, len(met) + 1), met, '-^', label=params[idx])
    
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
                  fontsize=legend_fontsize,
                  ncol=2)
        
    elif model == 'adversarial' and not re.search(r'loss', metric):
        ax.legend(fancybox=True,
                  shadow=True,
                  bbox_to_anchor=(1.02, 0.5),
                  ncol=1,
                  fontsize=legend_fontsize)
        
    elif task_setting == 'multi' and re.search(r'val', metric) and not re.search(r'loss', metric):
        ax.legend(fancybox=True,
                  shadow=True,
                  loc='lower right',
                  ncol=2,
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
    
    if model == 'adversarial' and aux == 'aux_1':
        plt.savefig('./plots/' + 'models/' + task + '/' + model + '/' + task_setting + '/' + aux + '/' + aux_task + '/' + layer
                    + '/' + task_sampling + '/' + metric + '.png')
    
    elif model == 'adversarial':
        plt.savefig('./plots/' + 'models/' + task + '/' + model + '/' + task_setting + '/' + aux + '/' + layer
                    + '/' + task_sampling + '/' + metric + '.png')
    
    elif model == 'adversarial' and correlation:
        plt.savefig('./plots/' + 'models/' + task + '/' + model + '/' + task_setting + '/'  + aux + '/' +  'train_vs_val' + '.png')
    
    elif model in ['SubjQA', 'combined', 'recurrent'] and task_setting == 'multi':
        plt.savefig('./plots/' + 'models/' + task + '/' + model + '/' + task_setting + '/' + layer + '/'
                    + task_sampling + '/' + metric + '.png')
        
    elif model == 'linear' and task_setting == 'multi' and aux == 'aux_1':
        plt.savefig('./plots/' + 'models/' + task + '/' + model + '/' + task_setting + '/' + aux + '/' + aux_task + '/'
                    + task_sampling + '/' + metric + '.png')
        
    elif model == 'linear' and task_setting == 'multi' and aux == 'aux_2':
        plt.savefig('./plots/' + 'models/' + task + '/' + model + '/' + task_setting + '/' + aux + '/'
                    + task_sampling + '/' + metric + '.png')
        
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
             layer=None,
             aux=None,
             aux_task=None,
             task_sampling=None,
):
    print('===========================')
    print('====== Task: {} ======'.format(task.upper()))
    print('===========================')
    print()
    for i, model in enumerate(models):
        print('==============================')
        print('====== Model: {} ======'.format(model.upper()))
        print('==============================')
        print()
        # load results
        all_results = get_results(
                                  task=task,
                                  version=version, 
                                  model=model,
                                  task_setting='multi' if model == 'adversarial' else task_setting, 
                                  layer=layer,
                                  aux=aux if model == 'adversarial' or model == 'linear' else None,
                                  aux_task=aux_task if (model == 'adversarial' or model == 'linear') and aux == 'aux_1' else None,
                                  task_sampling=task_sampling,
        )
        for j, metric in enumerate(metrics):
            print('=============================')
            print('===== Metric: {} ====='.format(metric.upper()))
            print('=============================')
            print()
            # plot results
            plot_results(
                         all_results, 
                         task=task, 
                         metric=metric, 
                         iv='methods' if model == 'adversarial' else iv,  
                         model=model, 
                         task_setting='multi' if model == 'adversarial' else task_setting, 
                         layer=layer, 
                         aux=aux if model == 'adversarial' or model == 'linear' else None,
                         aux_task=aux_task if (model == 'adversarial' or model == 'linear') and aux == 'aux_1' else None,
                         task_sampling=task_sampling,
            )
            
###########################################
################## t-SNE ##################
###########################################

def plot_seqs_projected_via_tsne(
                                 tsne_embed_x:np.ndarray,
                                 tsne_embed_y:np.ndarray,
                                 y_true:np.ndarray,
                                 class_to_idx:dict,
                                 model_name:str,
                                 task:str,
                                 combined_ds:bool=False,
                                 layer_wise:bool=False,
                                 n_layer:str=None,
                                 plot_qa:bool=False,
                                 sent_pair:list=None,
):
    plt.figure(figsize=(16,10), dpi=300) #NOTE: the higher the dpi the better the resolution
    ax = plt.subplot(111)
    
    dataset = '$D_{SubjQA} \: \cup \: D_{SQuAD}$' if combined_ds else 'D_{SubjQA}'
    
    # set hyperparameters
    legend_fontsize = 14
    title_fontsize = 16 
    
    # hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # only show ticks on the left (y-axis) and bottom (x-axis) spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
    # specify both labels for legend and colors for data points
    if len(np.unique(y_true)) > 3:
        classes = list(class_to_idx.keys())
    else:
        if plot_qa:
            classes = ['question', 'answer', 'context']
            markers = ['d', '*', 'o']
            colors = ['royalblue', 'firebrick', 'dimgrey']
        else:
            classes = ['$\mathbf{D}_{SubjQA}^{obj}$', '$\mathbf{D}_{SubjQA}^{sbj}$', '$\mathbf{D}_{SQuAD}$']
            colors = ['royalblue', 'palevioletred', 'green']

    for cat, lab in class_to_idx.items():
        if plot_qa:
            ax.scatter(
                       tsne_embed_x[y_true == lab],
                       tsne_embed_y[y_true == lab],
                       c=colors[lab],
                       marker=markers[lab],
                       alpha=.6,
                       label=classes[lab],
            )
        elif len(np.unique(y_true)) > 3:
            ax.scatter(
                       tsne_embed_x[y_true == lab],
                       tsne_embed_y[y_true == lab],
                       alpha=.6,
                       label=cat,
            )
        else:   
            ax.scatter(
                       tsne_embed_x[y_true == lab],
                       tsne_embed_y[y_true == lab],
                       color=colors[lab],
                       alpha=.6,
                       label=classes[lab],
            )
        
    if plot_qa:
        assert isinstance(sent_pair, list)
        special_toks = ['[CLS]', '[SEP]']
        for t, tok in enumerate(sent_pair):
            if tok not in special_toks:
                ax.annotate(tok, (tsne_embed_x[t], tsne_embed_y[t]))

    ax.legend(fancybox=True, shadow=True, loc='upper right', fontsize=legend_fontsize)
    
    if layer_wise:
        layer = n_layer.split('_')
        layer = ' '.join(layer).capitalize()
        ax.set_title('Model fine-tuned on' + ' ' + dataset + ':' + ' ' + layer, fontsize=title_fontsize)
        plt.tight_layout()
        plt.savefig('./plots/feat_reps/layer_wise/' + task + '/' + model_name + '_' + n_layer.lower() + '.png')
    else:
        ax.set_title('Model fine-tuned on' + ' ' + dataset, fontsize=title_fontsize)
        plt.tight_layout()
        plt.savefig('./plots/feat_reps/' + model_name + '.png')
    
    plt.show()
    plt.clf()
    

def plot_feat_reps_per_layer(
                             y_true:np.ndarray,
                             feat_reps_per_layer:dict,
                             class_to_idx:dict,
                             retained_variance:float,
                             rnd_state:int,
                             model_name:str,
                             task:str,
                             combined_ds:bool=False,
                             plot_qa:bool=False,
                             sent_pair:list=None,
):
    for layer, feat_reps in feat_reps_per_layer.items():
        # initiliase PCA
        pca = PCA(n_components=retained_variance, svd_solver='full', random_state=rnd_state)
        # project feat reps onto n principial components
        transformed_feats = pca.fit_transform(feat_reps)
        # initiliase TSNE
        tsne = TSNE(n_components=2, random_state=rnd_state)
        # transform projected feat reps via t-SNE
        tsne_embeds = tsne.fit_transform(transformed_feats)
        # get x_pos and y_pos of transformed data
        tsne_embed_x = tsne_embeds[:, 0]
        tsne_embed_y = tsne_embeds[:, 1]
        # plot model's feature representations at layer l in 2D space
        plot_seqs_projected_via_tsne(
                                     tsne_embed_x,
                                     tsne_embed_y,
                                     y_true,
                                     class_to_idx,
                                     model_name,
                                     task=task,
                                     combined_ds=combined_ds,
                                     layer_wise=True,
                                     n_layer=layer,
                                     plot_qa=plot_qa,
                                     sent_pair=sent_pair,
                                     )
    
##########################################
########## CONFUSION MATRIX ##############
##########################################
    
def conf_mat(
             y_pred:np.ndarray,
             y_true:np.ndarray,
             normalize:bool=False,
             metric=None,
):
    n = len(np.unique(y_true))
    conf_mat = np.zeros((n, n), dtype=int)
    for i, pred in enumerate(y_pred):
        conf_mat[y_true[i], pred] += 1
    
    if normalize:
        assert isinstance(metric, str), 'If normalized confusion matrix, metric must be defined'
        precision_scores = conf_mat.astype('float') / conf_mat.sum(axis=0)[:, np.newaxis]
        recall_scores = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        f1_scores = 2 * (precision_scores * recall_scores) / (precision_scores + recall_scores)
        
        if metric == 'precision':
            return precision_scores
        elif metric == 'recall':
            return recall_scores
        elif metric == 'f1':
            return f1_scores
        
    return conf_mat

def plot_confusion_matrix(
                          y_pred:np.ndarray,
                          y_true:np.ndarray,
                          labels:np.ndarray,
                          display_labels:list,
                          custom_conf_mat:bool,
                          normalize:bool,
                          metric=None,
                          sample_weight=None,
                          include_values=True,
                          xticks_rotation='horizontal',
                          values_format=None,
                          cmap='viridis',
                          ax=None,
                         ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if custom_conf_mat:
        if normalize:
            assert isinstance(metric, str), 'If normalized confusion matrix, metric must be defined'
        cm = conf_mat(
                      y_pred=y_pred,
                      y_true=y_true,
                      normalize=normalize,
                      metric=metric,
        )
    else:
        normalize = False
        cm = confusion_matrix(
                              y_true, 
                              y_pred, 
                              sample_weight=sample_weight,
                              labels=labels,
        )
        
    if display_labels is None:
        display_labels = labels

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=display_labels)
    return disp.plot(include_values=include_values,
                     cmap=cmap, ax=ax, xticks_rotation=xticks_rotation,
                     values_format=values_format)


class ConfusionMatrixDisplay:
    """Confusion Matrix visualization.
    It is recommend to use :func:`~sklearn.metrics.plot_confusion_matrix` to
    create a :class:`ConfusionMatrixDisplay`. All parameters are stored as
    attributes.
    Read more in the :ref:`User Guide <visualizations>`.
    Parameters
    ----------
    confusion_matrix : ndarray of shape (n_classes, n_classes)
        Confusion matrix.
    display_labels : ndarray of shape (n_classes,)
        Display labels for plot.
    Attributes
    ----------
    im_ : matplotlib AxesImage
        Image representing the confusion matrix.
    text_ : ndarray of shape (n_classes, n_classes), dtype=matplotlib Text, \
            or None
        Array of matplotlib axes. `None` if `include_values` is false.
    ax_ : matplotlib Axes
        Axes with confusion matrix.
    figure_ : matplotlib Figure
        Figure containing the confusion matrix.
    """
    def __init__(self, confusion_matrix, display_labels):
        self.confusion_matrix = confusion_matrix
        self.display_labels = display_labels

    def plot(self, include_values=True, cmap='viridis',
             xticks_rotation='horizontal', values_format=None, ax=None):
        """Plot visualization.
        Parameters
        ----------
        include_values : bool, default=True
            Includes values in confusion matrix.
        cmap : str or matplotlib Colormap, default='viridis'
            Colormap recognized by matplotlib.
        xticks_rotation : {'vertical', 'horizontal'} or float, \
                         default='horizontal'
            Rotation of xtick labels.
        values_format : str, default=None
            Format specification for values in confusion matrix. If `None`,
            the format specification is '.2g'.
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.
        Returns
        -------
        display : :class:`~sklearn.metrics.ConfusionMatrixDisplay`
        """
        check_matplotlib_support("ConfusionMatrixDisplay.plot")
        
        plt.figure(figsize=(14, 10), dpi=300)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        cm = self.confusion_matrix
        n_classes = cm.shape[0]
        self.im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        self.text_ = None

        cmap_min, cmap_max = self.im_.cmap(0), self.im_.cmap(256)

        if include_values:
            self.text_ = np.empty_like(cm, dtype=object)
            if values_format is None:
                values_format = '.2g'

            # print text with appropriate color depending on background
            thresh = (cm.max() + cm.min()) / 2.0
            for i, j in product(range(n_classes), range(n_classes)):
                color = cmap_max if cm[i, j] < thresh else cmap_min
                self.text_[i, j] = ax.text(j, i,
                                           format(cm[i, j], values_format),
                                           ha="center", va="center",
                                           color=color)

        fig.colorbar(self.im_, ax=ax)
        ax.set(xticks=np.arange(n_classes),
               yticks=np.arange(n_classes),
               xticklabels=self.display_labels,
               yticklabels=self.display_labels,
               ylabel="True label",
               xlabel="Predicted label")

        ax.set_ylim((n_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

        self.figure_ = fig
        self.ax_ = ax
        return self