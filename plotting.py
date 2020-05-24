__all__ = [
           'conf_mat',
           'plot_confusion_matrix',
           'plot_feat_reps_per_layer',
           'plot_reps_projected_via_tsne',
]

import numpy as np
import pandas as pd
import matplotlib as mpl 
import matplotlib.pyplot as plt
import seaborn as sns

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

from tqdm import trange, tqdm


###########################################
################## t-SNE ##################
###########################################

def plot_reps_projected_via_tsne(
                                 tsne_embed_x:np.ndarray,
                                 tsne_embed_y:np.ndarray,
                                 y_true:np.ndarray,
                                 class_to_idx:dict,
                                 file_name:str,
                                 source:str,
                                 version:str,
                                 combined_ds:bool=False,
                                 layer_wise:bool=False,
                                 n_layer:str=None,
                                 plot_qa:bool=False,
                                 sent_pair:list=None,
                                 support_labels=None,
                                 text_annotations:bool=False,
):
    #NOTE: uncomment line below if you want to use a dark background for plots
    #plt.style.use('dark_background')
    plt.figure(figsize=(16,10), dpi=300) #NOTE: the higher the dpi the better the resolution of the plot
    ax = plt.subplot(111)
    
    dataset = '$D_{SubjQA} \: \cup \: D_{SQuAD}$' if combined_ds else '$D_{SubjQA}$'
    
    #set hyperparameters
    legend_fontsize = 14
    title_fontsize = 16 
    
    #hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #only show ticks on the left (y-axis) and bottom (x-axis) spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
    # specify both labels for legend and colors for data points
    if len(np.unique(y_true)) > 3 and not plot_qa:
        legend_fontsize = 12
        classes = list(class_to_idx.keys())
        colors = ['royalblue', 'red', 'darkorange', 'indigo', 'hotpink', 'green']
        #NOTE: uncomment line below when using dark background
        #colors = ['royalblue', 'red', 'darkorange', 'cyan', 'hotpink', 'green'] 
        markers = ['o', 'd', '*', '+', '^', 'p']

        assert len(classes) == len(colors)
        #assert isinstance(support_labels, np.ndarray), 'Both context-domain and subjectivity labels must be provided'
        #assert support_labels.shape == y_true.shape, 'Shapes of context-domain and subjectivity labels vectors must be the same'
    else:
        if plot_qa:
            classes = list(class_to_idx.keys())
            markers = ['o', 'd', '*']
            colors = ['dimgrey', 'royalblue', 'firebrick']
        else:
            classes = ['$\mathbf{D}_{SubjQA}^{obj}$', '$\mathbf{D}_{SubjQA}^{sbj}$', '$\mathbf{D}_{SQuAD}$']
            colors = ['royalblue', 'palevioletred', 'green']

    for cat, lab in class_to_idx.items():
        if plot_qa:
            ax.scatter(
                       tsne_embed_x[y_true == lab],
                       tsne_embed_y[y_true == lab],
                       c = colors[lab],
                       marker = markers[lab],
                       alpha = .5 if lab == 0 else 1.0,
                       label = classes[lab],
            )
        elif len(np.unique(y_true)) > 3 and isinstance(support_labels, np.ndarray):
          for j, sbj_lab in enumerate(np.unique(support_labels)):
              ax.scatter(
                         tsne_embed_x[np.add(y_true, support_labels) == np.add(lab, sbj_lab)],
                         tsne_embed_y[np.add(y_true, support_labels) == np.add(lab, sbj_lab)],
                         marker = markers[lab],
                         color = colors[lab],
                         alpha = .9 if sbj_lab == 1 else .3, #display subjective questions with higher color intensity than objective questions
                         label = cat.capitalize() + ' ' + '(sbj)' if sbj_lab == 1 else cat.capitalize() + ' ' + '(obj)',
              )
        else:   
            ax.scatter(
                       tsne_embed_x[y_true == lab],
                       tsne_embed_y[y_true == lab],
                       color = colors[lab],
                       alpha = .6,
                       label = classes[lab],
            )
    
    if plot_qa:
        if text_annotations:
            assert isinstance(sent_pair, list)
            special_toks = ['[SEP]'] if y_true.tolist().index(class_to_idx['answer']) == 0 else ['[CLS]', '[SEP]']
            for t, tok in enumerate(sent_pair):
                if tok not in special_toks:
                    ax.annotate(tok, (tsne_embed_x[t], tsne_embed_y[t] + .5))
            subfolder = 'annotations/'
        else:
            subfolder = 'no_annotations/'
    else:
        subfolder = ''

    ax.legend(fancybox=True, shadow=True, loc='best', fontsize=legend_fontsize)
    
    PATH = './plots/hidden_reps/layer_wise/' + subfolder + source.lower() + '/' + version.lower() + '/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    layer = n_layer.split('_')
    layer = ' '.join(layer).capitalize()
    plt.tight_layout()
    plt.savefig(PATH + file_name + '_' + n_layer.lower() + '.png')
    plt.close()

def plot_feat_reps_per_layer(
                             y_true:np.ndarray,
                             feat_reps_per_layer:dict,
                             class_to_idx:dict,
                             retained_variance:float,
                             rnd_state:int,
                             file_name:str,
                             source:str,
                             version:str,
                             combined_ds:bool=False,
                             plot_qa:bool=False,
                             sent_pair:list=None,
                             support_labels=None,
                             text_annotations:bool=False,
):
    for layer, feat_reps in feat_reps_per_layer.items():
        #NOTE: uncomment line below, if you don't want to exploit t-SNE
        #pca = PCA(n_components=2, svd_solver='auto', random_state=rnd_state)
        pca = PCA(n_components=retained_variance, svd_solver='full', random_state=rnd_state)
        #project hidden reps onto n principal components that explain XY% of the original representation's variance
        transformed_feats = pca.fit_transform(feat_reps)
        #initiliase TSNE
        tsne = TSNE(n_components=2, random_state=rnd_state)
        #transform projected hidden reps via t-SNE into R^2
        tsne_embeds = tsne.fit_transform(transformed_feats)
        #get x_pos and y_pos of transformed data
        tsne_embed_x = tsne_embeds[:, 0]
        tsne_embed_y = tsne_embeds[:, 1]
        #plot model's hidden representations (in 2D space) for layer l 
        plot_reps_projected_via_tsne(
                                     tsne_embed_x,
                                     tsne_embed_y,
                                     y_true,
                                     class_to_idx,
                                     file_name,
                                     source=source,
                                     version=version,
                                     combined_ds=combined_ds,
                                     layer_wise=True,
                                     n_layer=layer,
                                     plot_qa=plot_qa,
                                     sent_pair=sent_pair,
                                     support_labels=support_labels,
                                     text_annotations=text_annotations,
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