__all__ = ['DistilBertForQA']

import numpy as np
import torch.nn as nn

import random
import torch

from transformers import DistilBertModel, DistilBertPreTrainedModel
from models.modules.QAHeads import *

# set random seeds to reproduce results
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DistilBertForQA(DistilBertPreTrainedModel):
    
    def __init__(
                 self,
                 config,
                 max_seq_length:int,
                 encoder:bool=False,
                 highway_connection:bool=False,
                 multitask:bool=False,
                 adversarial:bool=False,
                 dataset_agnostic:bool=False,
                 n_aux_tasks=None,
                 n_domain_labels=None,
                 n_qa_type_labels=None,
                 task:str='QA',
    ):        
        super(DistilBertForQA, self).__init__(config)
        self.distilbert = DistilBertModel(config)
        self.max_seq_length = max_seq_length
        self.encoder = encoder
        self.highway_connection = highway_connection
        self.multitask = multitask
        self.adversarial = adversarial
        self.dataset_agnostic = dataset_agnostic
        self.n_aux_tasks = n_aux_tasks
        self.n_domain_labels = n_domain_labels
        self.n_qa_type_labels = n_qa_type_labels
        self.task = task

        if self.multitask: assert isinstance(self.n_aux_tasks, int), "If MTL setting, number of auxiliary tasks must be defined"
        
        if self.encoder:
            self.qa_head = RecurrentQAHead(
                                           in_size=config.dim,
                                           n_labels_qa=2, #config.num_labels,
                                           qa_dropout_p=0.1, #config.qa_dropout,
                                           max_seq_length=self.max_seq_length,
                                           highway_block=self.highway_connection,
                                           multitask=self.multitask,
                                           n_aux_tasks=self.n_aux_tasks,
                                           n_domain_labels=self.n_domain_labels,
                                           n_qa_type_labels=self.n_qa_type_labels,
                                           adversarial=self.adversarial,
                                           dataset_agnostic=self.dataset_agnostic,
                                           task=self.task
                                           )
        else:
            self.qa_head = LinearQAHead(
                                        in_size=config.dim,
                                        n_labels_qa=2, #config.num_labels,
                                        qa_dropout_p=0.1, #config.qa_dropout,
                                        highway_block=self.highway_connection,
                                        multitask=self.multitask,
                                        n_aux_tasks=self.n_aux_tasks,
                                        n_domain_labels=self.n_domain_labels,
                                        n_qa_type_labels=self.n_qa_type_labels,
                                        adversarial=self.adversarial,
                                        dataset_agnostic=self.dataset_agnostic,
                                        task=self.task
                                        )
        self.init_weights()

    def forward(
                self,
                input_ids:torch.Tensor,
                attention_masks:torch.Tensor,
                token_type_ids:torch.Tensor,
                task:str,
                aux_targets=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                input_lengths=None,
                start_positions=None,
                end_positions=None,
                output_feat_reps:bool=False,
    ):
        #NOTE: token_type_ids == segment_ids (!)
        distilbert_output = self.distilbert(
                                        input_ids=input_ids,
                                        #token_type_ids=token_type_ids,
                                        attention_mask=attention_masks,
                                        head_mask=head_mask,
                                        )
      
        if self.encoder:
            return self.qa_head(
                                distilbert_output=distilbert_output,
                                seq_lengths=input_lengths,
                                task=task,
                                aux_targets=aux_targets,
                                start_positions=start_positions,
                                end_positions=end_positions,
                                output_feat_reps=output_feat_reps,
            )
        else:
            return self.qa_head(
                                distilbert_output=distilbert_output,
                                task=task,
                                aux_targets=aux_targets,
                                start_positions=start_positions,
                                end_positions=end_positions,
                                output_feat_reps=output_feat_reps,
            )