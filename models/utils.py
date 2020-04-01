__all__ = [
           'accuracy',
           'f1',
           'freeze_transformer_layers',
           'get_answers',
           'compute_exact_batch',
           'compute_f1_batch',
           'create_optimizer',
           'to_cpu',
           'train',
           'train_all',
           'val',
           'test',
]

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import random
import re
import torch
import transformers

from collections import Counter, defaultdict
from sklearn.metrics import f1_score
from tqdm import trange, tqdm
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertModel, BertForQuestionAnswering

from eval_squad import compute_exact, compute_f1

# set random seeds to reproduce results
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == "cuda":
  # if we are on GPU, set cuda random seeds 
  torch.cuda.manual_seed_all(42)

## NOTE: use this function in case we want to use a unidirectional LSTM (or GRU) instead of a BiLSTM ##
##       BERT feature representation sequences have to be reversed (special [CLS] token corresponds to semantic representation of sentence) ##
def reverse_sequences(batch:torch.Tensor):
    return torch.tensor(list(map(lambda feat_reps: feat_reps[::-1], batch)), dtype=torch.double).to(device)

def soft_to_hard(probas:torch.Tensor):
    return torch.tensor(list(map(lambda p: 1 if p >= 0.5 else 0, to_cpu(probas, detach=True))), dtype=torch.double)

def accuracy(probas:torch.Tensor, y_true:torch.Tensor, task:str):
    y_pred = soft_to_hard(probas) if task == 'binary' else torch.argmax(to_cpu(probas, to_numpy=False), dim=1) 
    y_true = y_true.type_as(y_pred)
    return (y_pred == to_cpu(y_true, to_numpy=False)).float().mean().item()

def f1(probas:torch.Tensor, y_true:torch.Tensor, task:str, avg:str='macro'):
    y_pred = soft_to_hard(probas) if task == 'binary' else torch.argmax(to_cpu(probas, detach=True, to_numpy=False), dim=1)
    return f1_score(to_cpu(y_true), y_pred.numpy(), average=avg)

def compute_acc_per_ds(results_per_ds:dict):
  return {ds: {q_type: 100 * (score['correct'] / score['freq']) for q_type, score in q_types.items()} for ds, q_types in results_per_ds.items()}

def get_detailed_scores(
                        probas:torch.Tensor,
                        y_true:torch.Tensor,
                        y_ds:torch.Tensor,
                        results_per_ds:dict,
                        ):
    y_pred = soft_to_hard(probas) 
    y_true = to_cpu(y_true.type_as(y_pred), to_numpy=False)
    y_ds = to_cpu(y_ds.type_as(y_pred), to_numpy=False)

    for p, l, ds in zip(y_pred, y_true, y_ds):
      try:
        results_per_ds['SubjQA' if ds == 1 else 'SQuAD']['sbj' if l == 1 else 'obj']['freq'] += 1
      except KeyError:
        results_per_ds['SubjQA' if ds == 1 else 'SQuAD']['sbj' if l == 1 else 'obj'] = {}
        results_per_ds['SubjQA' if ds == 1 else 'SQuAD']['sbj' if l == 1 else 'obj']['freq'] = 1
        results_per_ds['SubjQA' if ds == 1 else 'SQuAD']['sbj' if l == 1 else 'obj']['correct'] = 0
      if p == l:
        results_per_ds['SubjQA' if ds == 1 else 'SQuAD']['sbj' if l == 1 else 'obj']['correct'] += 1
    return results_per_ds

def freeze_transformer_layers(
                              model,
                              model_name:str,
                              unfreeze:bool,
                              l:int=None,
):
    model_names = ['bert', 'distilbert']
    model_name = model_name.lower()
    if model_name not in model_names:
        raise ValueError('Incorrect model name provided. Model name must be one of {}'.format(model_names))

    for n, p in model.named_parameters():
        if n.startswith(model_name):
            if unfreeze:
                assert isinstance(l, int)
                transformer_layer = model_name + '.transformer.layer.' if model_name == 'distilbert' else model_name + '.encoder.layer.' 
                pooling_layer = model_name + '.pooler.'
                if re.search(r'' + transformer_layer, n):
                    n_digits = '1' if model_name == 'distilbert' else '2'
                    if re.search(r'[0-9]' +'{' + n_digits + '}', n):
                        layer_no = n[len(transformer_layer): len(transformer_layer) + int(n_digits)]
                        if int(layer_no) >= l:
                            p.requires_grad = True
                elif re.search(r'' + pooling_layer, n):
                    p.requires_grad =True
            else:
                p.requires_grad = False
                
    return model

def get_answers(
                tokenizer,
                b_input_ids:torch.Tensor,
                start_logs:torch.Tensor,
                end_logs:torch.Tensor,
                predictions:bool,
):
    answers = []
    for input_ids, start_log, end_log in zip(b_input_ids, start_logs, end_logs):
        all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        if predictions:
            answer = ' '.join(all_tokens[torch.argmax(start_log):torch.argmax(end_log) + 1])
        else:
            answer = ' '.join(all_tokens[start_log:end_log + 1])
        answers.append(answer)
    return answers

def compute_exact_batch(
                        answers_gold:list,
                        answers_pred:list,
):
    return sum([compute_exact(a_gold, a_pred) for a_gold, a_pred in zip(answers_gold, answers_pred)])

def compute_f1_batch(
                     answers_gold:list,
                     answers_pred:list,
):
    return sum([compute_f1(a_gold, a_pred) for a_gold, a_pred in zip(answers_gold, answers_pred)])

# move tensor to CPU
def to_cpu(
           tensor:torch.Tensor,
           detach:bool=False,
           to_numpy:bool=True,
):
    tensor = tensor.detach().cpu() if detach else tensor.cpu()
    if to_numpy: return tensor.numpy()
    else: return tensor

def to_cat(
           true_labels:torch.Tensor,
           n_labels:int,
           ):
    batch_size = true_labels.size(0)
    cat_mat = torch.zeros(batch_size, n_labels)
    for i, l in enumerate(true_labels):
      cat_mat[i, l] += 1
    return cat_mat.to(device)

def create_optimizer(
                     model,
                     task:str,
                     eta:float,
                     ):
    task = task.lower()
    if task == 'qa':
        head = 'fc_qa'
    elif task == 'sbj_class':
        head = 'fc_sbj'
    elif task == 'domain_class':
        head = 'fc_domain'
    elif task == 'dataset_class':
        head = 'fc_ds'
    else:
        raise ValueError('Incorrect task name provided')
  
    no_decay = ["bias", "LayerNorm.weight"]
    optim_grouped_parameters = [
    {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and re.search(r'(' + 'bert' + '|' + head + ')', n)],
     "weight_decay": 0.0},
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and re.search(r'(' + 'bert' + '|' + head + ')', n)],
     "weight_decay": 0.0},
    ]
  
    optimizer = AdamW(
                      optim_grouped_parameters,
                      lr=eta, 
                      correct_bias=True,
    )
    return optimizer

def train(
          model,
          tokenizer,
          train_dl,
          val_dl,
          batch_size:int,
          args:dict,
          optimizer_qa,
          optimizer_sbj=None,
          optimizer_dom=None,
          optimizer_ds=None,
          scheduler_qa=None,
          scheduler_sbj=None,
          scheduler_dom=None,
          scheduler_ds=None,
          early_stopping:bool=True,
          n_aux_tasks=None,
          qa_type_weights=None,
          domain_weights=None,
          ds_weights=None,
          max_epochs:int=3,
          adversarial_simple:bool=False,
          multi_qa_type_class:bool=False,
          dataset_agnostic:bool=False,
          plot_task_distrib:bool=False,
):
    n_iters = args['n_steps'] * args['n_epochs']
    n_examples = args['n_steps'] * batch_size
    
    if args['n_evals'] == 'multiple_per_epoch':
      steps_until_eval =  args['n_steps'] // args['n_evals_per_epoch'] # number of steps between validations
      stop_training = False
    
    L = 6 # total number of transformer layers in DistilBERT model (L = 12 for BERT base, L = 6 for DistilBERT base)
    
    if args['freeze_bert']:
      k = int(L / (args['n_epochs'] - 1))
      l = L - k
      model_name = args['pretrained_model']
      model = freeze_transformer_layers(model, model_name=model_name, unfreeze=False)
      print("--------------------------------------------------")
      print("------ Pre-trained BERT weights are frozen -------")
      print("--------------------------------------------------")
      print()
        
    # keep track of batch losses, accuracies and F1s for plotting
    batch_losses = []
    val_losses = []
    val_accs = []
    val_f1s = []

    if args['task'] == 'QA':
      # define loss function (Cross-Entropy is numerically more stable than LogSoftmax plus Negative-Log-Likelihood Loss)
      qa_loss_func = nn.CrossEntropyLoss()
      tasks = ['QA']
      batch_accs_qa, batch_f1s_qa = [], []

      if isinstance(n_aux_tasks, int):
          tasks.append('Sbj_Class')
          if args['dataset'] == 'combined' and multi_qa_type_class:
            assert isinstance(qa_type_weights, torch.Tensor), 'Tensor of class weights for question-answer types is not provided'
            sbj_loss_func = nn.CrossEntropyLoss(weight=qa_type_weights.to(device))
          elif args['dataset'] == 'combined':
            assert isinstance(qa_type_weights, torch.Tensor), 'Tensor of class weights for question-answer types is not provided'
            sbj_loss_func = nn.BCEWithLogitsLoss(pos_weight=qa_type_weights.to(device))
          else:
            sbj_loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.ones(2).to(device))

          batch_accs_sbj, batch_f1s_sbj = [], []

          if n_aux_tasks == 2 and dataset_agnostic:
              assert isinstance(ds_weights, torch.Tensor), 'Tensor of class weights for the two datasets is not provided'
              ds_loss_func = nn.BCEWithLogitsLoss(weight=ds_weights.to(device))
              batch_accs_ds, batch_f1s_ds = [], []
              tasks.append('Dataset_Class')
        
          elif n_aux_tasks == 2 and not dataset_agnostic:
              assert isinstance(domain_weights, torch.Tensor), 'Tensor of class weights for different domains is not provided'
              domain_loss_func = nn.CrossEntropyLoss(weight=domain_weights.to(device))
              batch_accs_domain, batch_f1s_domain = [], []
              tasks.append('Domain_Class')

              if args['mtl_setting'] == 'domain_only':
                tasks.pop(tasks.index('Sbj_Class'))

      loss_func = qa_loss_func

    elif args['task'] == 'Sbj_Classification':
      tasks = ['Sbj_Class']
      if args['dataset'] == 'combined' and multi_qa_type_class:
        assert isinstance(qa_type_weights, torch.Tensor), 'Tensor of class weights for question-answer types is not provided'
        sbj_loss_func = nn.CrossEntropyLoss(weight=qa_type_weights.to(device))
      elif args['dataset'] == 'combined':
        assert isinstance(qa_type_weights, torch.Tensor), 'Tensor of class weights for question-answer types is not provided'
        sbj_loss_func = nn.BCEWithLogitsLoss(pos_weight=qa_type_weights.to(device))
      else:
        sbj_loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.ones(2).to(device))

      batch_accs_sbj, batch_f1s_sbj = [], []
      loss_func = sbj_loss_func

    elif args['task'] == 'Domain_Classification':
      tasks = ['Domain_Class']
      assert isinstance(domain_weights, torch.Tensor), 'Tensor of class weights for different domains is not provided'
      domain_loss_func = nn.CrossEntropyLoss(weight=domain_weights.to(device))

      batch_accs_domain, batch_f1s_domain = [], []
      loss_func = domain_loss_func

    if isinstance(n_aux_tasks, type(None)) or args['task_sampling'] == 'uniform' or args['task'] in ['Sbj_Classification', 'Domain_Classification']:
      distrib = [1/len(tasks) for _ in tasks]
      
    elif isinstance(n_aux_tasks, int) and args['task_sampling'] == 'oversampling':
      distrib = [2/3 if task == 'QA' else 1/(3 * (len(tasks) - 1)) for task in tasks]

    task_order = np.random.choice(tasks, size=args['n_steps'], replace=True, p = distrib)
    task_distrib = Counter(task_order)

    if plot_task_distrib:
      plt.bar(tasks, [task_distrib[task] for task in tasks], alpha=0.5, edgecolor='black')
      plt.xticks(range(len(tasks)), labels=tasks)
      plt.xlabel('Tasks', fontsize=12)
      plt.ylabel('Frequency per epoch', fontsize=12)
      plt.title('Task distribution in MTL setting')
      plt.show()
      plt.clf()

    # we want to store train exact-match accuracies and F1 scores for each task as often as we evaluate model on validation set
    running_tasks = tasks[:]

    for epoch in trange(args['n_epochs'],  desc="Epoch"):

        ### Training ###

        model.train()

        if args['task'] == 'QA':
          correct_answers, batch_f1 = 0, 0

          if isinstance(n_aux_tasks, int):
            if 'Sbj_Class' in tasks:
              batch_acc_sbj, batch_f1_sbj = 0, 0

            if 'Dataset_Class' in tasks:
              batch_acc_ds, batch_f1_ds = 0, 0
            
            if 'Domain_Class' in tasks:
              batch_acc_domain, batch_f1_domain = 0, 0

        elif args['task'] == 'Sbj_Classification':
          batch_acc_sbj, batch_f1_sbj = 0, 0

        elif args['task'] == 'Domain_Classification':
          batch_acc_domain, batch_f1_domain = 0, 0

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        
        # n_steps == n_updates per epoch (n_iters = n_epochs * n_steps per epoch)
        for step, batch in enumerate(tqdm(train_dl, desc="Step")):

            if args['batch_presentation'] == 'alternating' and isinstance(n_aux_tasks, int):
              assert len(batch) == 2, 'In MTL, we must provide batches with different input sequences for the main and auxiliary task when alternating'
              main_batch = tuple(t.to(device) for t in batch[0])
              aux_sbj_batch = tuple(t.to(device) for t in batch[1])
            else:
              main_batch = tuple(t.to(device) for t in batch)
            
            # sample task from random distribution
            current_task = task_order[step]

            # set loss back to 0 after every training iteration
            batch_loss = 0 
            
            if isinstance(n_aux_tasks, int):
              print('------------------------------------')
              print('-------- Current task: {} --------'.format(current_task))
              print('------------------------------------')
              print()

            if current_task == 'QA':

              ######################################
              ###### QUESTION ANSWERING TASK #######
              ######################################

              b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, b_start_pos, b_end_pos, _, _, _ = main_batch

              start_logits, end_logits = model(
                             input_ids=b_input_ids,
                             attention_masks=b_attn_masks,
                             token_type_ids=b_token_type_ids,
                             input_lengths=b_input_lengths,
                             task=current_task,
                             )

              # start and end loss must be computed separately and then averaged
              start_loss = qa_loss_func(start_logits, b_start_pos)
              end_loss = qa_loss_func(end_logits, b_end_pos)
              batch_loss += (start_loss + end_loss) / 2

              start_log_probas = to_cpu(F.log_softmax(start_logits, dim=1), detach=False, to_numpy=False)
              end_log_probas = to_cpu(F.log_softmax(end_logits, dim=1), detach=False, to_numpy=False)
            
              pred_answers = get_answers(
                                         tokenizer=tokenizer,
                                         b_input_ids=b_input_ids,
                                         start_logs=start_log_probas,
                                         end_logs=end_log_probas,
                                         predictions=True,
              )
            
              true_answers = get_answers(
                                         tokenizer=tokenizer,
                                         b_input_ids=b_input_ids,
                                         start_logs=b_start_pos,
                                         end_logs=b_end_pos,
                                         predictions=False,
              )
            
              correct_answers += compute_exact_batch(true_answers, pred_answers)
              batch_f1 += compute_f1_batch(true_answers, pred_answers)

              # keep track of train examples used for QA
              nb_tr_examples_qa = Counter(task_order[:step+1])[current_task] * batch_size

              current_batch_acc = round(100 * (correct_answers / nb_tr_examples_qa), 3)
              current_batch_f1 = round(100 * (batch_f1 / nb_tr_examples_qa), 3)
              
              print("--------------------------------------------")
              print("----- Current batch {} exact-match: {} % -----".format(current_task, current_batch_acc))
              print("----- Current batch {} F1: {} % -----".format(current_task, current_batch_f1))
              print("--------------------------------------------")
              print()

              if step > (steps_until_eval // 2):
                if current_task in running_tasks:
                  batch_accs_qa.append(current_batch_acc)
                  batch_f1s_qa.append(current_batch_f1)
                  running_tasks.pop(running_tasks.index(current_task))

            else:
              if current_task == 'Sbj_Class':
                if args['task'] == 'QA' and args['batch_presentation'] == 'alternating':
                  b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, b_sbj = aux_sbj_batch

                elif args['task'] == 'Sbj_Classification' and args['batch_presentation'] == 'alternating':
                  b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, b_sbj = main_batch

                else:
                  b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, _, _, b_sbj, _, _ = main_batch


                if multi_qa_type_class:

                  ##########################################################################
                  ##### MULTI-WAY SEQUENCE CLASSIFICATION OF QA TYPE (ONLY QUESTIONS) ######
                  ##########################################################################

                  sbj_logits = model(
                                     input_ids=b_input_ids,
                                     attention_masks=b_attn_masks,
                                     token_type_ids=b_token_type_ids,
                                     input_lengths=b_input_lengths,
                                     task=current_task,
                                     )

                  if adversarial_simple:
                    batch_loss -= sbj_loss_func(sbj_logits, b_sbj)
                  else:
                    batch_loss += sbj_loss_func(sbj_logits, b_sbj)
  
                  batch_acc_sbj += accuracy(probas=F.log_softmax(sbj_logits, dim=1), y_true=b_sbj, task='multi-way')  
                  batch_f1_sbj += f1(probas=F.log_softmax(sbj_logits, dim=1), y_true=b_sbj, task='multi-way')

                  batch_acc_aux = batch_acc_sbj
                  batch_f1_aux = batch_f1_sbj

                else:

                  ######################################################################
                  ##### BINARY SEQUENCE CLASSIFICATION OF BOTH ANSWERS & QUESTIONS #####
                  ######################################################################

                  sbj_logits_a, sbj_logits_q = model(
                                                     input_ids=b_input_ids,
                                                     attention_masks=b_attn_masks,
                                                     token_type_ids=b_token_type_ids,
                                                     input_lengths=b_input_lengths,
                                                     task=current_task,
                                                     )

                  sbj_logits = torch.stack((sbj_logits_a, sbj_logits_q), dim=1)
                        
                  b_sbj = b_sbj.type_as(sbj_logits)

                  if adversarial_simple:
                    batch_loss -= sbj_loss_func(sbj_logits, b_sbj)

                  else:
                    batch_loss += sbj_loss_func(sbj_logits, b_sbj)
      
                  current_sbj_acc = 0
                  current_sbj_f1 = 0

                  for k in range(b_sbj.size(1)):

                    current_sbj_acc += accuracy(probas=torch.sigmoid(sbj_logits[:, k]), y_true=b_sbj[:, k], task='binary')  
                    current_sbj_f1 += f1(probas=torch.sigmoid(sbj_logits[:, k]), y_true=b_sbj[:, k], task='binary')

                  batch_acc_sbj += (current_sbj_acc / b_sbj.size(1))
                  batch_f1_sbj += (current_sbj_f1 / b_sbj.size(1))

                  batch_acc_aux = batch_acc_sbj
                  batch_f1_aux = batch_f1_sbj

              elif current_task == 'Domain_Class':

                ##########################################################################
                ##### MULTI-WAY SEQUENCE CLASSIFICATION OF RESPECTIVE REVIEW DOMAINS #####
                ##########################################################################

                b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, _, _, _, b_domains, _ = main_batch

                domain_logits = model(
                                      input_ids=b_input_ids,
                                      attention_masks=b_attn_masks,
                                      token_type_ids=b_token_type_ids,
                                      input_lengths=b_input_lengths,
                                      task=current_task,
                  )

                if adversarial_simple:
                  batch_loss -= domain_loss_func(domain_logits, b_domains)

                else:
                  batch_loss += domain_loss_func(domain_logits, b_domains)

                batch_acc_domain += accuracy(probas=F.log_softmax(domain_logits, dim=1), y_true=b_domains, task='multi-way')  
                batch_f1_domain += f1(probas=F.log_softmax(domain_logits, dim=1), y_true=b_domains, task='multi-way')

                batch_acc_aux = batch_acc_domain
                batch_f1_aux = batch_f1_domain

              elif current_task == 'Dataset_Class':

                ################################################################################
                ##### BINARY SEQUENCE CLASSIFICATION OF DATASETS (i.e., SQUAD vs. SUBJQA ) #####
                ###################### NOTE: THIS IS AN ADVERSARIAL TASK #######################
                ################################################################################

                b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, _, _, _, _, b_ds = main_batch

                ds_logits = model(
                                  input_ids=b_input_ids,
                                  attention_masks=b_attn_masks,
                                  token_type_ids=b_token_type_ids,
                                  input_lengths=b_input_lengths,
                                  task=current_task,
                  )

                b_ds = b_ds.type_as(ds_logits)

                if adversarial_simple:
                  batch_loss -= ds_loss_func(ds_logits, b_ds)

                else:
                  batch_loss += ds_loss_func(ds_logits, b_ds)

                  batch_acc_ds += accuracy(probas=torch.sigmoid(ds_logits), y_true=b_ds, task='binary')  
                  batch_f1_ds += f1(probas=torch.sigmoid(ds_logits), y_true=b_ds, task='binary')

                batch_acc_aux = batch_acc_ds
                batch_f1_aux = batch_f1_ds
              
              # keep track of steps taken per task (don't use overall steps)
              nb_tr_steps_aux = Counter(task_order[:step+1])[current_task]
              current_batch_acc_aux = round(100 * (batch_acc_aux / nb_tr_steps_aux), 3)
              current_batch_f1_aux = round(100 * (batch_f1_aux / nb_tr_steps_aux), 3)

              print("--------------------------------------------")
              print("----- Current batch {} acc: {} % -----".format(current_task, current_batch_acc_aux))
              print("----- Current batch {} F1: {} % -----".format(current_task, current_batch_f1_aux))
              print("--------------------------------------------")
              print()

              # we don't want to save F1 scores and exact-match accuracies at the very beginning of training
              if step > (steps_until_eval // 2):
                if current_task in running_tasks:
                  if current_task == 'Sbj_Class':
                    batch_accs_sbj.append(current_batch_acc_aux)
                    batch_f1s_sbj.append(current_batch_f1_aux)

                  elif current_task == 'Domain_Class':
                    batch_accs_domain.append(current_batch_acc_aux)
                    batch_f1s_domain.append(current_batch_f1_aux)

                  elif current_task == 'Dataset_Class':
                    batch_accs_ds.append(current_batch_acc_aux)
                    batch_f1s_ds.append(current_batch_f1_aux)

                  running_tasks.pop(running_tasks.index(current_task))

            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

            print("------------------------------------")
            print("----- Current {} loss: {} -----".format(current_task, abs(round(batch_loss.item(), 3))))
            print("------------------------------------")
            print()

            # in any MTL setting, we exclusively want to store QA losses (there's no need to store losses for auxiliary tasks since we want to observe effect on main task)
            if isinstance(n_aux_tasks, int):
              if current_task == 'QA':
                tr_loss += batch_loss.item()
                batch_losses.append(batch_loss.item())
            else:
                tr_loss += batch_loss.item()
                batch_losses.append(batch_loss.item())

            batch_loss.backward()
            
            # clip gradients if gradients become larger than predefined gradient norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), args["max_grad_norm"])

            # take step down the valley w.r.t. current task
            if current_task == 'QA':
              optimizer_qa.step()
              scheduler_qa.step()
              optimizer_qa.zero_grad()
            
            elif current_task == 'Sbj_Class':
              optimizer_sbj.step()
              if not isinstance(scheduler_sbj, type(None)):
                scheduler_sbj.step()
              optimizer_sbj.zero_grad()

            elif current_task == 'Dataset_Class':
              optimizer_ds.step()
              if not isinstance(scheduler_ds, type(None)):
                scheduler_ds.step()
              optimizer_ds.zero_grad()

            elif current_task == 'Domain_Class':
              optimizer_dom.step()
              if not isinstance(scheduler_dom, type(None)):
                scheduler_dom.step()
              optimizer_dom.zero_grad()

            if args['n_evals'] == 'multiple_per_epoch':
              if step > 0 and step % steps_until_eval == 0:
                val_losses, val_accs, val_f1s, model = val(
                                                          model=model,
                                                          tokenizer=tokenizer,
                                                          val_dl=val_dl,
                                                          args=args,
                                                          current_step=step,
                                                          epoch=epoch,
                                                          batch_size=batch_size,
                                                          val_losses=val_losses,
                                                          val_accs=val_accs,
                                                          val_f1s=val_f1s,
                                                          loss_func=loss_func,
                                                          multi_qa_type_class=multi_qa_type_class,
                                                          )

                # we want to store train exact-match accuracies and F1 scores for each task as often as we evaluate model on validation set
                running_tasks = tasks[:]
                  
                # after evaluation on dev set, move model back to train mode
                model.train()

                # we want to train the model at least for one epoch
                if epoch > 0 and early_stopping:
                  # if loss has not decreased for the past args['early_stopping_thresh'] eval steps, stop training
                  if np.argmin(val_losses[::-1]) > args['early_stopping_thresh']:
                    stop_training = True
                    break

        if args['task'] == 'QA':
          tr_loss /= task_distrib['QA']
        elif args['task'] == 'Sbj_Classification': 
          tr_loss /= task_distrib['Sbj_Class']
        elif args['task'] == 'Domain_Classification': 
          tr_loss /= task_distrib['Domain_Class']

        print("------------------------------------")
        print("---------- EPOCH {} ----------".format(epoch + 1))
        print("----- Train loss: {} -----".format(round(tr_loss, 3)))

        if args['task'] == 'QA':
          train_exact_match = round(100 * (correct_answers / (task_distrib['QA'] * batch_size)), 3)
          train_f1 = round(100 * (batch_f1 / (task_distrib['QA'] * batch_size)), 3)
          print("----- Train QA exact-match: {} % -----".format(round(train_exact_match, 3)))
          print("----- Train QA F1: {} % -----".format(round(train_f1, 3)))

          if isinstance(n_aux_tasks, int) and (len(batch_accs_sbj) > 0 and len(batch_f1s_sbj) > 0):

             print("------------------------------------")
             print("----- Train sbj acc: {} % -----".format(batch_accs_sbj[-1]))
             print("----- Train sbj F1: {} % -----".format(batch_f1s_sbj[-1]))
             print("------------------------------------")
             print()

             if n_aux_tasks == 2 and dataset_agnostic:

                print("------------------------------------")
                print("----- Train dataset acc: {} % -----".format(batch_accs_ds[-1]))
                print("----- Train dataset F1: {} % -----".format(batch_f1s_ds[-1]))
                print("------------------------------------")
                print()

             elif n_aux_tasks == 2 and not dataset_agnostic:  
                print("------------------------------------")
                print("----- Train domain acc: {} % -----".format(batch_accs_domain[-1]))
                print("----- Train domain F1: {} % -----".format(batch_f1s_domain[-1]))
                print("------------------------------------")
                print()

        elif args['task'] == 'Sbj_Classification':
          print("----- Train Sbj acc: {} % -----".format(batch_accs_sbj[-1]))
          print("----- Train Sbj F1: {} % -----".format(batch_f1s_sbj[-1]))

        elif args['task'] == 'Domain_Classification':
          print("----- Train Domain acc: {} % -----".format(batch_accs_domain[-1]))
          print("----- Train Domain F1: {} % -----".format(batch_f1s_domain[-1]))

        print("----------------------------------")
        print()
        
        if args['n_evals'] == 'one_per_epoch':
          val_losses, val_accs, val_f1s, model = val(
                                                    model=model,
                                                    tokenizer=tokenizer,
                                                    val_dl=val_dl,
                                                    args=args,
                                                    current_step=step,
                                                    epoch=epoch,
                                                    batch_size=batch_size,
                                                    val_losses=val_losses,
                                                    val_accs=val_accs,
                                                    val_f1s=val_f1s,
                                                    loss_func=loss_func,
                                                    multi_qa_type_class=multi_qa_type_class,
                                                    )

          # we want to store train exact-match accuracies and F1 scores for each task as often as we evaluate model on validation set
          running_tasks = tasks[:]

          # after evaluation on dev set, move model back to train mode
          model.train()

          if epoch > 0 and early_stopping:
            if val_losses[-1] > val_losses[-2]:
              print("------------------------------------------")
              print("----- Early stopping after {} steps -----".format(nb_tr_steps + len(train_dl) * epoch))
              print("------------------------------------------")
              break
        else:
          if stop_training:
            print("------------------------------------------")
            print("----- Early stopping after {} steps -----".format(nb_tr_steps + len(train_dl) * epoch))
            print("------------------------------------------")
            break

    # return model in eval mode
    model.eval()
    if isinstance(n_aux_tasks, type(None)) and args['task'] == 'QA':
      return batch_losses, batch_accs_qa, batch_f1s_qa, val_losses, val_accs, val_f1s, model
    elif isinstance(n_aux_tasks, type(None)) and args['task'] == 'Sbj_Classification':
      return batch_losses, batch_accs_sbj, batch_f1s_sbj, val_losses, val_accs, val_f1s, model
    elif isinstance(n_aux_tasks, type(None)) and args['task'] == 'Domain_Classification':
      return batch_losses, batch_accs_domain, batch_f1s_domain, val_losses, val_accs, val_f1s, model
    elif n_aux_tasks == 1:
      return batch_losses, batch_accs_qa, batch_f1s_qa, batch_accs_sbj, batch_f1s_sbj, val_losses, val_accs, val_f1s, model
    elif n_aux_tasks == 2 and dataset_agnostic:
      return batch_losses, batch_accs_qa, batch_f1s_qa, batch_accs_sbj, batch_f1s_sbj, batch_accs_ds, batch_f1s_ds, val_losses, val_accs, val_f1s, model
    elif n_aux_tasks == 2 and not dataset_agnostic:
      return batch_losses, batch_accs_qa, batch_f1s_qa, batch_accs_sbj, batch_f1s_sbj, batch_accs_domain, batch_f1s_domain, val_losses, val_accs, val_f1s, model

def val(
        model,
        tokenizer,
        val_dl,
        args:dict,
        current_step:int,
        epoch:int,
        batch_size:int,
        val_losses:list,
        val_accs:list,
        val_f1s:list,
        loss_func=None,
        sequential_transfer:bool=False,
        evaluation_strategy:str=None,
        multi_qa_type_class:bool=False,
):
    ### Validation ###

    # set model to eval mode
    model.eval()

    # n_features in DistilBERT transformer layers
    distilbert_hidden_size = 768

    # path to save models
    model_path = args['model_dir'] 
    
    if args['task'] == 'QA':
      correct_answers_val = 0

    elif args['task'] == 'Sbj_Classification':
      batch_acc_sbj = 0

    elif args['task'] == 'Domain_Classification':
      batch_acc_domain = 0
    
    batch_f1_val = 0
    val_loss = 0
    nb_val_steps, nb_val_examples = 0, 0

    for batch in val_dl:
        # move every tensor in batch to current device
        batch = tuple(t.to(device) for t in batch)
        
        batch_loss_val = 0
        
        ### UNPACK INPUTS FROM MINI-BATCH FOR CURRENT TASK ###

        if args['task'] == 'QA' and sequential_transfer:
          b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, b_start_pos, b_end_pos, b_sbj, b_domains, _ = batch

        elif args['task'] == 'QA' and not sequential_transfer:
          b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, b_start_pos, b_end_pos, _, _, _ = batch

        elif args['task'] == 'Sbj_Classification':
          if args['batch_presentation'] == 'alternating':
            b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, b_sbj = batch
          else:
            b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, _, _, b_sbj, _, _ = batch
            
        elif args['task'] == 'Domain_Classification':
          b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, _, _,  _, b_domains, _ = batch

        else:
          raise ValueError('Incorrect task name provided')

        ##########################################################

        # if number of examples in current batch is smaller than specified batch_size, skip batch
        if b_input_ids.size(0) != batch_size:
            continue
        
        # no gradient computation in evaluation mode
        with torch.no_grad():
          if args['task'] == 'QA':
              if sequential_transfer:
                  if evaluation_strategy == 'oracle':
                      # inform model about true labels corresponding to auxiliary tasks
                      one_hot_domains = to_cat(true_labels=b_domains, n_labels=args['n_domains'])
                      b_sbj = b_sbj.type_as(one_hot_domains)
                      b_aux_hard_targets = torch.cat((b_sbj, one_hot_domains), dim=1)

                      # perform QA task with hard targets from both auxiliary tasks as additional information about (q, c) sequence pair
                      ans_logits_val = model(
                                             input_ids=b_input_ids,
                                             attention_masks=b_attn_masks,
                                             token_type_ids=b_token_type_ids,
                                             input_lengths=b_input_lengths,
                                             task='QA',
                                             aux_targets=b_aux_hard_targets,
                      )
                  elif evaluation_strategy == 'soft_targets':
                      # perform subjectivity classification task
                      sbj_logits_a, sbj_logits_q = model( 
                                                         input_ids=b_input_ids,
                                                         attention_masks=b_attn_masks,
                                                         token_type_ids=b_token_type_ids,
                                                         input_lengths=b_input_lengths,
                                                         task='Sbj_Class',
                                                        )
                              
                      # pass model's raw (sbj) output logits through sigmoid function to yield probability scores
                      sbj_probas = torch.stack((torch.sigmoid(sbj_logits_a), torch.sigmoid(sbj_logits_q)), dim=1)

                      # perform context-domain classification task
                      domain_logits = model(
                                            input_ids=b_input_ids,
                                            attention_masks=b_attn_masks,
                                            token_type_ids=b_token_type_ids,
                                            input_lengths=b_input_lengths,
                                            task='Domain_Class',
                                            )
                      # pass model's raw (context-domain) output logits through softmax function to yield probability distribution over domain classes 
                      soft_domains = F.softmax(domain_logits, dim=1)

                      # create mini-batch of soft targets for both auxiliary tasks
                      b_aux_soft_targets = torch.cat((sbj_probas, soft_domains), dim=1)

                      # perform QA task with soft targets from both auxiliary tasks as additional information about (q, c) sequence pair
                      ans_logits_val = model(
                                            input_ids=b_input_ids,
                                            attention_masks=b_attn_masks,
                                            token_type_ids=b_token_type_ids,
                                            input_lengths=b_input_lengths,
                                            task='QA',
                                            aux_targets=b_aux_soft_targets,
                                            )
                  elif evaluation_strategy == 'no_aux_targets':

                      ### make sure model does not constain weights for auxiliary task probability scores ###

                      try:
                        assert model.qa_head.fc_qa.weight.size(1) == distilbert_hidden_size
                      except AssertionError:
                        with torch.no_grad():
                          model.qa_head.fc_qa.weight = nn.Parameter(model.qa_head.fc_qa.weight[:, :distilbert_hidden_size])
                          model.qa_head.fc_qa.in_features = distilbert_hidden_size

                      ########################################################################################

                      # perform QA task without any additional information about auxiliary tasks at evaluation time
                      ans_logits_val = model(
                                             input_ids=b_input_ids,
                                             attention_masks=b_attn_masks,
                                             token_type_ids=b_token_type_ids,
                                             input_lengths=b_input_lengths,
                                             task='QA',
                                             )
              else:
                  ans_logits_val = model(
                                         input_ids=b_input_ids,
                                         attention_masks=b_attn_masks,
                                         token_type_ids=b_token_type_ids,
                                         input_lengths=b_input_lengths,
                                         task='QA'
                                         )

              start_logits_val, end_logits_val = ans_logits_val

              start_true_val = to_cpu(b_start_pos)
              end_true_val = to_cpu(b_end_pos)
              
              # start and end loss must be computed separately
              start_loss = loss_func(start_logits_val, b_start_pos)
              end_loss = loss_func(end_logits_val, b_end_pos)
              batch_loss_val = (start_loss + end_loss) / 2
              
              print("----------------------------------------")
              print("----- Current val batch loss: {} -----".format(round(batch_loss_val.item(), 3)))
              print("----------------------------------------")
              print()
              
              start_log_probs_val = to_cpu(F.log_softmax(start_logits_val, dim=1), detach=True, to_numpy=False)
              end_log_probs_val = to_cpu(F.log_softmax(end_logits_val, dim=1), detach=True, to_numpy=False)
          
              pred_answers = get_answers(
                                         tokenizer=tokenizer,
                                         b_input_ids=b_input_ids,
                                         start_logs=start_log_probs_val,
                                         end_logs=end_log_probs_val,
                                         predictions=True,
              )

              true_answers = get_answers(
                                         tokenizer=tokenizer,
                                         b_input_ids=b_input_ids,
                                         start_logs=b_start_pos,
                                         end_logs=b_end_pos,
                                         predictions=False,
              )
              
              correct_answers_val += compute_exact_batch(true_answers, pred_answers)
              batch_f1_val += compute_f1_batch(true_answers, pred_answers)

          elif args['task'] == 'Sbj_Classification':

              if multi_qa_type_class:
                  sbj_logits = model(
                                     input_ids=b_input_ids,
                                     attention_masks=b_attn_masks,
                                     token_type_ids=b_token_type_ids,
                                     input_lengths=b_input_lengths,
                                     task='Sbj_Class',
                                     )

                  batch_loss_val += loss_func(sbj_logits, b_sbj)

                  batch_acc_sbj += accuracy(probas=F.log_softmax(sbj_logits, dim=1), y_true=b_sbj, task='multi-way')  
                  batch_f1_val += f1(probas=F.log_softmax(sbj_logits, dim=1), y_true=b_sbj, task='multi-way')

              else:
                  sbj_logits_a, sbj_logits_q = model(
                                                     input_ids=b_input_ids,
                                                     attention_masks=b_attn_masks,
                                                     token_type_ids=b_token_type_ids,
                                                     input_lengths=b_input_lengths,
                                                     task='Sbj_Class',
                                                     )

                  sbj_logits = torch.stack((sbj_logits_a, sbj_logits_q), dim=1)
                  
                  b_sbj = b_sbj.type_as(sbj_logits)

                  batch_loss_val += loss_func(sbj_logits, b_sbj)

                  current_sbj_acc = 0
                  current_sbj_f1 = 0

                  for k in range(b_sbj.size(1)):

                    current_sbj_acc += accuracy(probas=torch.sigmoid(sbj_logits[:, k]), y_true=b_sbj[:, k], task='binary')  
                    current_sbj_f1 += f1(probas=torch.sigmoid(sbj_logits[:, k]), y_true=b_sbj[:, k], task='binary')

                  batch_acc_sbj += (current_sbj_acc / b_sbj.size(1))
                  batch_f1_val += (current_sbj_f1 / b_sbj.size(1))

          elif args['task'] == 'Domain_Classification':

            domain_logits = model(
                                  input_ids=b_input_ids,
                                  attention_masks=b_attn_masks,
                                  token_type_ids=b_token_type_ids,
                                  input_lengths=b_input_lengths,
                                  task='Domain_Class',
                  )


            batch_loss_val += loss_func(domain_logits, b_domains)

            batch_acc_domain += accuracy(probas=F.log_softmax(domain_logits, dim=1), y_true=b_domains, task='multi-way')  
            batch_f1_val += f1(probas=F.log_softmax(domain_logits, dim=1), y_true=b_domains, task='multi-way')

          print("----------------------------------------")
          print("----- Current val batch loss: {} -----".format(round(batch_loss_val.item(), 3)))
          print("----------------------------------------")
          print()

          val_loss += batch_loss_val.item()
          nb_val_examples += b_input_ids.size(0)
          nb_val_steps += 1

          current_batch_f1 = 100 * (batch_f1_val / nb_val_examples) if args['task'] == 'QA' else 100 * (batch_f1_val / nb_val_steps )

          if args['task'] == 'QA':
            current_batch_acc = 100 * (correct_answers_val / nb_val_examples)

          elif args['task'] == 'Sbj_Classification':
            current_batch_acc = 100 * (batch_acc_sbj / nb_val_steps)

          elif args['task'] == 'Domain_Classification':
            current_batch_acc = 100 * (batch_acc_domain / nb_val_steps)

    val_loss /= nb_val_steps
    print("----------------------------------")
    print("-------- Train step {} --------".format(current_step + 1))
    print("----- Val loss: {} -----".format(round(val_loss, 3)))

    if args['task'] == 'QA':
      val_exact_match = 100 * (correct_answers_val / nb_val_examples)
      val_f1 = 100 * (batch_f1_val / nb_val_examples)

      print("----- Val QA exact-match: {} % -----".format(round(val_exact_match, 3)))
      print("----- Val QA F1: {} % -----".format(round(val_f1, 3)))
    
    elif args['task'] == 'Sbj_Classification':
      val_acc = 100 * (batch_acc_sbj / nb_val_steps)
      val_f1 = 100 * (batch_f1_val / nb_val_steps)

      print("----- Val Sbj acc: {} % -----".format(round(val_acc, 3)))
      print("----- Val Sbj F1: {} % -----".format(round(val_f1, 3)))

    elif args['task'] == 'Domain_Classification':
      val_acc = 100 * (batch_acc_domain / nb_val_steps)
      val_f1 = 100 * (batch_f1_val / nb_val_steps)

      print("----- Val Domain acc: {} % -----".format(round(val_acc, 3)))
      print("----- Val Domain F1: {} % -----".format(round(val_f1, 3)))

    if epoch == 0 or val_loss < min(val_losses):
      torch.save(model.state_dict(), model_path + '/%s' % (args['model_name']))

    print("----------------------------------")
    print()

    val_losses.append(val_loss)
    val_accs.append(val_exact_match if args['task'] == 'QA' else val_acc)
    val_f1s.append(val_f1)

    return val_losses, val_accs, val_f1s, model

def test(
        model,
        tokenizer,
        test_dl,
        batch_size:int,
        not_finetuned:bool=False,
        task:str='QA',
        n_domains:int=6,
        input_sequence:str='question_context',
        sequential_transfer:bool=False,
        inference_strategy:str=None,
        detailed_analysis_sbj_class:bool=False,
        multi_qa_type_class:bool=False,

):
    n_steps = len(test_dl)
    n_examples = n_steps * batch_size
    distilbert_hidden_size = 768
    
    #################
    ### Inference ###
    #################

    model.eval()

    ### INITIALISE LOSS FUNCTION FOR RESPECTIVE TASK ###

    if task == 'QA':
      correct_answers_test = 0
      loss_func = nn.CrossEntropyLoss()

    elif task == 'Sbj_Classification':
      batch_acc_test = 0
      if multi_qa_type_class:
        loss_func = nn.CrossEntropyLoss()
        predictions, true_labels, feat_reps = [], [], []
      else:
        loss_func = nn.BCEWithLogitsLoss()

    elif task == 'Domain_Classification':
      batch_acc_test = 0
      loss_func = nn.CrossEntropyLoss()

    ###################################################

    ######### DETAILED ANALYSIS ###########

    if detailed_analysis_sbj_class:
      results_per_ds = defaultdict(dict)

    ########################################
    
    batch_f1_test = 0
    test_f1, test_loss = 0, 0
    nb_test_steps, nb_test_examples = 0, 0

    for n, batch in enumerate(test_dl):
        # move all tensors in batch to current device (e.g., GPU)
        batch = tuple(t.to(device) for t in batch)

        batch_loss_test = 0
        
        ### UNPACK INPUTS FROM MINI-BATCH FOR RESPECTIVE TASK ###

        if task == 'QA' and sequential_transfer:
          b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, b_start_pos, b_end_pos, b_sbj, b_domains, _ = batch

        elif task == 'QA' and not sequential_transfer:
          b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, b_start_pos, b_end_pos, _, _, _ = batch

        elif task == 'Sbj_Classification':

          if detailed_analysis_sbj_class:
            b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, _, _, b_sbj, _, b_ds = batch

          elif input_sequence == 'question_context':
            b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, _, _, b_sbj, _, _ = batch

          elif input_sequence == 'question_answer':
            b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, b_sbj, _ = batch

        elif task == 'Domain_Classification':
          b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, _, _,  _, b_domains, _ = batch

        else:
          raise ValueError('Incorrect task name provided')

        ##########################################################

        ## NOTE: number of examples in last mini-batch might not be equal to batch_size ##
        ## if the latter is the case and number of examples in current batch is smaller than specified batch_size, skip batch ##
        if b_input_ids.size(0) != batch_size:
            continue

        with torch.no_grad():
            if task == 'QA':
              if not_finetuned:
                  start_logits_test, end_logits_test = model(
                                                             input_ids=b_input_ids,
                                                             attention_mask=b_attn_masks,
                  )
              else:
                if sequential_transfer:
                  if inference_strategy == 'oracle':
                    # inform model about true labels for auxiliary tasks
                    one_hot_domains = to_cat(true_labels=b_domains, n_labels=n_domains)
                    b_sbj = b_sbj.type_as(one_hot_domains)
                    b_aux_hard_targets = torch.cat((b_sbj, one_hot_domains), dim=1)

                    # perform QA task with hard targets from both auxiliary tasks as additional information about question-context sequence pair
                    start_logits_test, end_logits_test = model(
                                                               input_ids=b_input_ids,
                                                               attention_masks=b_attn_masks,
                                                               token_type_ids=b_token_type_ids,
                                                               input_lengths=b_input_lengths,
                                                               task='QA',
                                                               aux_targets=b_aux_hard_targets,
                    )
                  elif inference_strategy == 'soft_targets':
                      # perform subjectivity classification task
                      sbj_logits_a, sbj_logits_q = model( 
                                                         input_ids=b_input_ids,
                                                         attention_masks=b_attn_masks,
                                                         token_type_ids=b_token_type_ids,
                                                         input_lengths=b_input_lengths,
                                                         task='Sbj_Class',
                                                        )
                        
                      # pass model's raw (sbj) output logits through sigmoid function to yield probability scores
                      sbj_probas = torch.stack((torch.sigmoid(sbj_logits_a), torch.sigmoid(sbj_logits_q)), dim=1)

                      # perform context-domain classification task
                      domain_logits = model(
                                            input_ids=b_input_ids,
                                            attention_masks=b_attn_masks,
                                            token_type_ids=b_token_type_ids,
                                            input_lengths=b_input_lengths,
                                            task='Domain_Class',
                                            )
                      # pass model's raw (context-domain) output logits through softmax function to yield probability distribution over domain classes 
                      soft_domains = F.softmax(domain_logits, dim=1)

                      # create mini-batch of soft targets for both auxiliary tasks
                      b_aux_soft_targets = torch.cat((sbj_probas, soft_domains), dim=1)

                      # perform QA task with soft targets from both auxiliary tasks as additional information about question-context sequence pair
                      start_logits_test, end_logits_test = model(
                                                                input_ids=b_input_ids,
                                                                attention_masks=b_attn_masks,
                                                                token_type_ids=b_token_type_ids,
                                                                input_lengths=b_input_lengths,
                                                                task='QA',
                                                                aux_targets=b_aux_soft_targets,
                                                                )
                  elif inference_strategy == 'no_aux_targets':

                      ### make sure model does not constain weights for soft targets from auxiliary tasks ###

                      if n == 0:
                          try:
                              assert model.qa_head.fc_qa.weight.size(1) == distilbert_hidden_size
                          except AssertionError:
                              model.qa_head.fc_qa.weight = nn.Parameter(model.qa_head.fc_qa.weight[:, :distilbert_hidden_size])
                              model.qa_head.fc_qa.in_features = distilbert_hidden_size

                      ########################################################################################

                      # perform QA task without any additional information about auxiliary tasks at test time
                      start_logits_test, end_logits_test = model(
                                                                 input_ids=b_input_ids,
                                                                 attention_masks=b_attn_masks,
                                                                 token_type_ids=b_token_type_ids,
                                                                 input_lengths=b_input_lengths,
                                                                 task='QA',
                                                                 )
                  else:
                    raise ValueError('Incorrect name for inference strategy in sequential transfer setting provided.')
                else:
                    start_logits_test, end_logits_test = model(
                                                               input_ids=b_input_ids,
                                                               attention_masks=b_attn_masks,
                                                               token_type_ids=b_token_type_ids,
                                                               input_lengths=b_input_lengths,
                                                               task='QA',
                                                               )

              # move true start and end positions of answer span to CPU
              start_true_test = to_cpu(b_start_pos)
              end_true_test = to_cpu(b_end_pos)

              # start and end loss must be computed separately
              start_loss = loss_func(start_logits_test, b_start_pos)
              end_loss = loss_func(end_logits_test, b_end_pos)

              batch_loss_test = (start_loss + end_loss) / 2

              start_log_probs_test = to_cpu(F.log_softmax(start_logits_test, dim=1), detach=True, to_numpy=False)
              end_log_probs_test = to_cpu(F.log_softmax(end_logits_test, dim=1), detach=True, to_numpy=False)

              pred_answers = get_answers(
                                         tokenizer=tokenizer,
                                         b_input_ids=b_input_ids,
                                         start_logs=start_log_probs_test,
                                         end_logs=end_log_probs_test,
                                         predictions=True,
              )

              true_answers = get_answers(
                                         tokenizer=tokenizer,
                                         b_input_ids=b_input_ids,
                                         start_logs=b_start_pos,
                                         end_logs=b_end_pos,
                                         predictions=False,
              )

              correct_answers_test += compute_exact_batch(true_answers, pred_answers)
              batch_f1_test += compute_f1_batch(true_answers, pred_answers)

            elif task == 'Sbj_Classification':

              if multi_qa_type_class:
                  sbj_logits, feat_reps_cls = model(
                                                input_ids=b_input_ids,
                                                attention_masks=b_attn_masks,
                                                token_type_ids=b_token_type_ids,
                                                input_lengths=b_input_lengths,
                                                task='Sbj_Class',
                                                output_feat_reps=True,
                                                )

                  batch_loss_test += loss_func(sbj_logits, b_sbj)

                  sbj_log_probas = F.log_softmax(sbj_logits, dim=1)

                  batch_acc_test += accuracy(probas=sbj_log_probas, y_true=b_sbj, task='multi-way')  
                  batch_f1_test += f1(probas=sbj_log_probas, y_true=b_sbj, task='multi-way')

                  #### STORE FEATURE REPRESENTATIONS PER LABEL & MODEL'S PREDICTIONS ####

                  y_hat_q_type = torch.argmax(to_cpu(sbj_log_probas, detach=True, to_numpy=False), dim=1).numpy().tolist()
                  y_true = to_cpu(b_sbj, detach=False, to_numpy=True).tolist()
                  feat_reps_cls = to_cpu(feat_reps_cls, detach=True, to_numpy=True).tolist()

                  predictions.append(y_hat_q_type)
                  true_labels.append(y_true)
                  
                  for feat_rep in feat_reps_cls:
                    feat_reps.append(feat_rep)

              else:
                  sbj_logits_a, sbj_logits_q = model(
                                                     input_ids=b_input_ids,
                                                     attention_masks=b_attn_masks,
                                                     token_type_ids=b_token_type_ids,
                                                     input_lengths=b_input_lengths,
                                                     task='Sbj_Class',
                                                     )

                  sbj_logits = torch.stack((sbj_logits_a, sbj_logits_q), dim=1)
                  
                  b_sbj = b_sbj.type_as(sbj_logits)

                  batch_loss_test += loss_func(sbj_logits, b_sbj)

                  current_sbj_acc = 0
                  current_sbj_f1 = 0

                  for k in range(b_sbj.size(1)):

                    current_sbj_acc += accuracy(probas=torch.sigmoid(sbj_logits[:, k]), y_true=b_sbj[:, k], task='binary')  
                    current_sbj_f1 += f1(probas=torch.sigmoid(sbj_logits[:, k]), y_true=b_sbj[:, k], task='binary')

                    if detailed_analysis_sbj_class:
                      results_per_ds = get_detailed_scores(probas=torch.sigmoid(sbj_logits[:, k]), y_true=b_sbj[:, k], y_ds=b_ds, results_per_ds=results_per_ds)

                  batch_acc_test += (current_sbj_acc / b_sbj.size(1))
                  batch_f1_test += (current_sbj_f1 / b_sbj.size(1))

            elif task == 'Domain_Classification':

              domain_logits = model(
                                    input_ids=b_input_ids,
                                    attention_masks=b_attn_masks,
                                    token_type_ids=b_token_type_ids,
                                    input_lengths=b_input_lengths,
                                    task='Domain_Class',
                    )


              batch_loss_test += loss_func(domain_logits, b_domains)

              batch_acc_test += accuracy(probas=F.log_softmax(domain_logits, dim=1), y_true=b_domains, task='multi-way')  
              batch_f1_test += f1(probas=F.log_softmax(domain_logits, dim=1), y_true=b_domains, task='multi-way')

            test_loss += batch_loss_test.item()
            nb_test_examples += b_input_ids.size(0)
            nb_test_steps += 1
            
            current_batch_f1 = 100 * (batch_f1_test / nb_test_examples) if task == 'QA' else 100 * (batch_f1_test / nb_test_steps)
            current_batch_acc = 100 * (correct_answers_test / nb_test_examples) if task == 'QA' else 100 *(batch_acc_test / nb_test_steps)

            print("--------------------------------------------")
            print("----- Current batch exact-match: {} % -----".format(round(current_batch_acc, 3)))
            print("----- Current batch F1: {} % -----".format(round(current_batch_f1, 3)))
            print("--------------------------------------------")
            print()

    test_loss /= nb_test_steps

    print("-----------------------------------")
    print("------------ Inference ------------")
    print("------- Test loss: {} -------".format(round(test_loss, 3)))

    if task == 'QA':
      
      test_acc = 100 * (correct_answers_test / nb_test_examples)
      test_f1 = 100 * (batch_f1_test / nb_test_examples)

      print("----- Test QA exact-match: {} % -----".format(round(test_acc, 3)))
      print("----- Test QA F1: {} % -----".format(round(test_f1, 3)))
    
    else:

      test_acc = 100 * (batch_acc_test / nb_test_steps)
      test_f1 = 100 * (batch_f1_test / nb_test_steps)

      if task == 'Sbj_Classification':

        print("----- Test Sbj acc: {} % -----".format(round(test_acc, 3)))
        print("----- Test Sbj F1: {} % -----".format(round(test_f1, 3)))

      else:

        print("----- Test Domain acc: {} % -----".format(round(test_acc, 3)))
        print("----- Test Domain F1: {} % -----".format(round(test_f1, 3)))

    print("----------------------------------")
    print()

    if detailed_analysis_sbj_class:
      results_per_ds = compute_acc_per_ds(results_per_ds)
      return test_loss, test_acc, test_f1, results_per_ds
    if task == 'Sbj_Classification' and multi_qa_type_class:
      predictions = np.array(predictions).flatten().tolist()
      true_labels = np.array(true_labels).flatten().tolist()
      return test_loss, test_acc, test_f1, predictions, true_labels, feat_reps
    else:
      return test_loss, test_acc, test_f1

def train_all(
              model,
              tokenizer,
              train_dl,
              val_dl,
              batch_size:int,
              args:dict,
              train_dl_sbj=None,
              val_dl_sbj=None,
              early_stopping:bool=True,
              qa_type_weights=None,
              domain_weights=None,
              max_epochs:int=4,
              adversarial_simple:bool=False,
):
    n_iters = args['n_steps'] * args['n_epochs']
    n_examples = args['n_steps'] * batch_size

    # make sure, we fine-tune for max. 3 epochs (last epoch is eval round to store model's output logits for aux tasks)
    if args['training_regime'] == 'soft_targets':
      try:
          assert args['n_epochs'] == max_epochs
      except AssertionError:
          args['n_epochs'] = max_epochs
  
    if args['n_evals'] == 'multiple_per_epoch':
        steps_until_eval =  args['n_steps'] // args['n_evals_per_epoch'] # number of steps between validations

    # freeze transformer layers for debugging on local machine
    if args['freeze_bert']:
        model_name = args['pretrained_model']
        model = freeze_transformer_layers(model, model_name=model_name, unfreeze=False)
        print("--------------------------------------------------")
        print("------ Pre-trained BERT weights are frozen -------")
        print("--------------------------------------------------")
        print()

    # keep track of batch losses, accuracies and F1s for plotting
    batch_losses = []
    val_losses_all_tasks = []
    val_accs_all_tasks = []
    val_f1s_all_tasks = []

    qa_loss_func = nn.CrossEntropyLoss()
    batch_accs_qa, batch_f1s_qa = [], []

    if args['dataset'] == 'combined':
        assert isinstance(qa_type_weights, torch.Tensor), 'Tensor of class weights for question-answer types is not provided'
        print("Weights for subjective Anwers: {}".format(qa_type_weights[0]))
        print()
        print("Weights for subjective Questions: {}".format(qa_type_weights[1]))
        print()
        sbj_loss_func = nn.BCEWithLogitsLoss(pos_weight=qa_type_weights.to(device))
    else:
        sbj_loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.ones(2).to(device))

    batch_accs_sbj, batch_f1s_sbj = [], []
      
    assert isinstance(domain_weights, torch.Tensor), 'Tensor of class weights for different domains is not provided'
    domain_loss_func = nn.CrossEntropyLoss(weight=domain_weights.to(device))
    batch_accs_domain, batch_f1s_domain = [], []

    tasks = ['Domain_Class', 'Sbj_Class', 'QA']
    running_tasks = tasks[:]
  
    loss_funcs = [domain_loss_func, sbj_loss_func, qa_loss_func]
  
    if args['batch_presentation'] == 'alternating':
        # create copy of data loaders, when using (q, a) instead of (q, c) sequence pairs for sbj classification
        train_dl_copy = train_dl
        val_dl_copy = val_dl
  
    sbj_logits_all = []
    domain_logits_all = []

    for i, task in enumerate(tqdm(tasks, desc="Task")):
        
        ################################################ SEQUENTIAL TRANSFER ########################################################
        ############## Fine-tune model on every task sequentially (i.e., sequential transfer / soft-parameter sharing) ##############
        #### SOFT TARGETS: store model's auxiliary output logits for each mini-batch of input sequences after model convergence #####
        ###### ORACLE: concatenate true labels for auxiliary tasks with each contextual word embedding in any (q, c) sequence #######
        #############################################################################################################################

        #TODO: figure out, whether this is the correct way to modify input_size and weights of a fc (output) layer on the fly
        if task == 'QA':
            if args['training_regime'] == 'soft_targets':
                add_features = sbj_logits_all[0].size(1)
                if len(domain_logits_all) > 0:
                    add_features += domain_logits_all[0].size(1)

            elif args['training_regime'] == 'oracle':
                add_features = args['n_qa_labels'] + args['n_domains']

            with torch.no_grad():
                model.qa_head.fc_qa.in_features += add_features
                assert model.qa_head.fc_qa.out_features == args['n_qa_labels']
                model.qa_head.fc_qa.weight = nn.Parameter(torch.cat((model.qa_head.fc_qa.weight,
                                                                     torch.randn(add_features, args['n_qa_labels']).T.to(device)), 1))

        # initialize task-specific optimizers on the fly
        optimizer = create_optimizer(model=model, task=task, eta=args['lr_adam'])

        if i > 0:
            scheduler = get_linear_schedule_with_warmup(
                                                        optimizer, 
                                                        num_warmup_steps=args['warmup_steps'], 
                                                        num_training_steps=args['t_total'],
                                                        )

        eval_round = False
        stop_training = False

        print('------------------------------------')
        print('-------- Current task: {} --------'.format(task))
        print('------------------------------------')
        print()

        loss_func = loss_funcs[i]

        # make sure we fine-tune model on every task
        model.train()

        if task == 'Sbj_Class':
            if args['batch_presentation'] == 'alternating':
                assert not isinstance(train_dl_sbj, type(None)), 'If class. (q, a) in T_sbj, provide separate train dl for sbj. class.'
                assert not isinstance(val_dl_sbj, type(None)), 'If class. (q, a) in T_sbj, provide separate val dl for sbj. class.'
                train_dl = train_dl_sbj
                val_dl = val_dl_sbj
        else:
            if args['batch_presentation'] == 'alternating':
                train_dl = train_dl_copy
                val_dl = val_dl_copy

        val_losses = []
        val_accs = []
        val_f1s = []

        for j, epoch in enumerate(trange(args['n_epochs'],  desc="Epoch")):
            
            if task == 'QA':
                args['task'] = task
                correct_answers, batch_f1 = 0, 0
            
            elif task == 'Sbj_Class':
                args['task'] = 'Sbj_Classification'
                batch_acc_sbj, batch_f1_sbj = 0, 0
            
            elif task == 'Domain_Class':
                args['task'] = 'Domain_Classification'
                batch_acc_domain, batch_f1_domain = 0, 0

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            # n_steps == n_updates per epoch (n_iters = n_epochs * n_steps per epoch)
            for step, batch in enumerate(tqdm(train_dl, desc="Step")):
                
                batch = tuple(t.to(device) for t in batch)
        
                # set loss back to 0 after every iteration
                batch_loss = 0                       

                if task == 'QA':
                    # make sure, we are not in eval mode when fine-tuning on QA
                    assert eval_round == False

                    # unpack inputs from dataloader for main task           
                    b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, b_start_pos, b_end_pos, _, _, _ = batch

                    if args['training_regime'] == 'oracle':
                        # inform model about true labels corresponding to auxiliary tasks
                        one_hot_domains = to_cat(true_labels=b_domains, n_labels=args['n_domains'])
                        b_sbj = b_sbj.type_as(one_hot_domains)
                        b_aux_hard_targets = torch.cat((b_sbj, one_hot_domains), dim=1)

                        # perform QA task with hard targets from both auxiliary tasks as additional information about any (q, c) sequence pair
                        start_logits, end_logits = model(
                                                         input_ids=b_input_ids,
                                                         attention_masks=b_attn_masks,
                                                         token_type_ids=b_token_type_ids,
                                                         input_lengths=b_input_lengths,
                                                         task=task,
                                                         aux_targets=b_aux_hard_targets,
                                                         )

                    elif args['training_regime'] == 'soft_targets':
                        b_sbj_scores = sbj_logits_all[step]
                        b_soft_domains = domain_logits_all[step]
                        b_aux_soft_targets = torch.cat((b_sbj_scores, b_soft_domains), dim=1)

                        # perform QA task with soft targets from both auxiliary tasks as additional information about any (q, c) sequence pair
                        start_logits, end_logits = model(
                                                         input_ids=b_input_ids,
                                                         attention_masks=b_attn_masks,
                                                         token_type_ids=b_token_type_ids,
                                                         input_lengths=b_input_lengths,
                                                         task=task,
                                                         aux_targets=b_aux_soft_targets,
                        )

                    # start and end loss must be computed separately and then averaged
                    start_loss = loss_func(start_logits, b_start_pos)
                    end_loss = loss_func(end_logits, b_end_pos)
                    batch_loss += (start_loss + end_loss) / 2

                    start_log_probas = to_cpu(F.log_softmax(start_logits, dim=1), detach=False, to_numpy=False)
                    end_log_probas = to_cpu(F.log_softmax(end_logits, dim=1), detach=False, to_numpy=False)

                    pred_answers = get_answers(
                                               tokenizer=tokenizer,
                                               b_input_ids=b_input_ids,
                                               start_logs=start_log_probas,
                                               end_logs=end_log_probas,
                                               predictions=True,
                    )

                    true_answers = get_answers(
                                               tokenizer=tokenizer,
                                               b_input_ids=b_input_ids,
                                               start_logs=b_start_pos,
                                               end_logs=b_end_pos,
                                               predictions=False,
                    )

                    correct_answers += compute_exact_batch(true_answers, pred_answers)
                    batch_f1 += compute_f1_batch(true_answers, pred_answers)

                    nb_tr_examples += b_input_ids.size(0)
                    nb_tr_steps += 1

                    current_batch_acc = round(100 * (correct_answers / nb_tr_examples), 3)
                    current_batch_f1 = round(100 * (batch_f1 / nb_tr_examples), 3)

                    print("--------------------------------------------")
                    print("----- Current batch {} exact-match: {} % -----".format(task, current_batch_acc))
                    print("----- Current batch {} F1: {} % -----".format(task, current_batch_f1))
                    print("--------------------------------------------")
                    print()

                    if step > (steps_until_eval // 2):
                        if task in running_tasks:
                            batch_accs_qa.append(current_batch_acc)
                            batch_f1s_qa.append(current_batch_f1)
                            running_tasks.pop(running_tasks.index(task))

                else:
                    if task == 'Sbj_Class':
                        if args['batch_presentation'] == 'alternating':
                            # unpack inputs from data loader for (q, a) sequence pair inputs
                            b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, b_sbj = batch
                        else:
                            # unpack inputs from main data loader for (q, c) sequence pair inputs
                            b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, _, _, b_sbj, _, _ = batch

                        if eval_round:
                            # no gradient calculations in eval mode to speed up computation
                            with torch.no_grad():
                                # perform subjectivity classification task
                                sbj_logits_a, sbj_logits_q = model(
                                                                   input_ids=b_input_ids,
                                                                   attention_masks=b_attn_masks,
                                                                   token_type_ids=b_token_type_ids,
                                                                   input_lengths=b_input_lengths,
                                                                   task=task,
                                                                   )
                                # pass model's raw output logits through sigmoid function
                                # store probability scores for each input sequence in mini-batch
                                sbj_logits_all.append(torch.stack((torch.sigmoid(sbj_logits_a), torch.sigmoid(sbj_logits_q)), dim=1))
                        else:
                            # perform subjectivity classification task
                            sbj_logits_a, sbj_logits_q = model( 
                                                               input_ids=b_input_ids,
                                                               attention_masks=b_attn_masks,
                                                               token_type_ids=b_token_type_ids,
                                                               input_lengths=b_input_lengths,
                                                               task=task,
                            )
                            sbj_logits = torch.stack((sbj_logits_a, sbj_logits_q), dim=1)
                            b_sbj = b_sbj.type_as(sbj_logits)

                            if adversarial_simple:
                                batch_loss -= loss_func(sbj_logits, b_sbj)
                            else:
                                batch_loss += loss_func(sbj_logits, b_sbj)
                                
                            current_sbj_acc = 0
                            current_sbj_f1 = 0
                            
                            for k in range(b_sbj.size(1)):
                                current_sbj_acc += accuracy(probas=torch.sigmoid(sbj_logits[:, k]), y_true=b_sbj[:, k], task='binary')  
                                current_sbj_f1 += f1(probas=torch.sigmoid(sbj_logits[:, k]), y_true=b_sbj[:, k], task='binary')

                            batch_acc_sbj += (current_sbj_acc / b_sbj.size(1))
                            batch_f1_sbj += (current_sbj_f1 / b_sbj.size(1))

                            batch_acc_aux = batch_acc_sbj
                            batch_f1_aux = batch_f1_sbj

                    elif task == 'Domain_Class':
                        # unpack inputs from main data loader to perform context-domain classification on (q, c) sequence pairs
                        b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, _, _, _, b_domains, _ = batch

                        if eval_round:
                            # no gradient calculations in eval mode to speed up computation for storing model's predictions
                            with torch.no_grad():
                                # perform context-domain classification task
                                domain_logits = model(
                                                      input_ids=b_input_ids,
                                                      attention_masks=b_attn_masks,
                                                      token_type_ids=b_token_type_ids,
                                                      input_lengths=b_input_lengths,
                                                      task=task,
                                  )
                                # pass model's raw output logits through softmax function 
                                # to yield probability distribution over classes and store those probability scores for each input sequence in mini-batch
                                domain_logits_all.append(F.softmax(domain_logits, dim=1))
                        else:
                            # perform context-domain classification task
                            domain_logits = model(
                                                  input_ids=b_input_ids,
                                                  attention_masks=b_attn_masks,
                                                  token_type_ids=b_token_type_ids,
                                                  input_lengths=b_input_lengths,
                                                  task=task,
                            )

                            if adversarial_simple:
                                batch_loss -= loss_func(domain_logits, b_domains)
                            else:
                                batch_loss += loss_func(domain_logits, b_domains)
                                
                            batch_acc_domain += accuracy(probas=F.log_softmax(domain_logits, dim=1), y_true=b_domains, task='multi-way')
                            batch_f1_domain += f1(probas=F.log_softmax(domain_logits, dim=1), y_true=b_domains, task='multi-way')

                            batch_acc_aux = batch_acc_domain
                            batch_f1_aux = batch_f1_domain
          
                    if not eval_round:
                        nb_tr_examples += b_input_ids.size(0)
                        nb_tr_steps += 1
                        
                        current_batch_acc_aux = round(100 * (batch_acc_aux / nb_tr_steps), 3)
                        current_batch_f1_aux = round(100 * (batch_f1_aux / nb_tr_steps), 3)

                        print("--------------------------------------------")
                        print("----- Current batch {} acc: {} % -----".format(task, current_batch_acc_aux))
                        print("----- Current batch {} F1: {} % -----".format(task, current_batch_f1_aux))
                        print("--------------------------------------------")
                        print()

                        # we don't want to save F1 scores and exact-match accuracies at the very beginning of training
                        if step > (steps_until_eval // 2):
                            if task in running_tasks:
                                if task == 'Sbj_Class':
                                    batch_accs_sbj.append(current_batch_acc_aux)
                                    batch_f1s_sbj.append(current_batch_f1_aux)

                                elif task == 'Domain_Class':
                                    batch_accs_domain.append(current_batch_acc_aux)
                                    batch_f1s_domain.append(current_batch_f1_aux)
                                    
                                running_tasks.pop(running_tasks.index(task))

                if not eval_round:
                    print("------------------------------------")
                    print("----- Current {} loss: {} -----".format(task, abs(round(batch_loss.item(), 3))))
                    print("------------------------------------")
                    print()

                    ## in any MTL setting, we exclusively want to store QA losses
                    ## there's no need to store losses for auxiliary tasks since we want to observe effect of sequential transfer on main task
                    if task == 'QA':
                        tr_loss += batch_loss.item()
                        batch_losses.append(batch_loss.item())
                        
                    # backpropagate error
                    batch_loss.backward()
                    
                    # clip gradients if gradients become larger than predefined gradient norm
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args["max_grad_norm"])

                    # take step down the valley w.r.t. current task
                    optimizer.step()

                    # decrease learning rate linearly for all tasks but the first
                    if i > 0:
                        scheduler.step()

                    # after each training step, zero-out gradients
                    optimizer.zero_grad()

                    if args['n_evals'] == 'multiple_per_epoch':
                        if step > 0 and step % steps_until_eval == 0:
                            if task == 'QA' and args['evaluation_strategy'] == 'no_aux_targets':
                                # save model's current weights for aux targets (in current setting, we evaluate model without information about aux targets)
                                current_aux_weights = model.qa_head.fc_qa.weight[:, 768:]
                            
                            val_losses, val_accs, val_f1s, model = val(
                                                                       model=model,
                                                                       tokenizer=tokenizer,
                                                                       val_dl=val_dl,
                                                                       args=args,
                                                                       current_step=step,
                                                                       epoch=epoch,
                                                                       batch_size=batch_size,
                                                                       val_losses=val_losses,
                                                                       val_accs=val_accs,
                                                                       val_f1s=val_f1s,
                                                                       loss_func=loss_func,
                                                                       sequential_transfer=True,
                                                                       evaluation_strategy=args['evaluation_strategy'],

                            )
                            if task == 'QA' and args['evaluation_strategy'] == 'no_aux_targets':
                                with torch.no_grad():
                                    model.qa_head.fc_qa.in_features += add_features
                                    model.qa_head.fc_qa.weight = nn.Parameter(torch.cat((model.qa_head.fc_qa.weight, current_aux_weights.to(device)), 1))

                            # we want to store train exact-match accuracies and F1 scores for each task
                            # as often as we evaluate model on validation set
                            running_tasks = tasks[:]
                
                            # after evaluation on dev set, set model back to train mode
                            model.train()

                            # we want to train the model at least for one epoch
                            if epoch > 0 and early_stopping:
                                # if loss has not decreased for the past args['early_stopping_thresh'] eval steps, stop training early
                                if np.argmin(val_losses[::-1]) > args['early_stopping_thresh']:
                                    stop_training = True
                                    break

            if not eval_round:
                tr_loss /= nb_tr_steps
                print("------------------------------------")
                print("---------- EPOCH {} ----------".format(epoch + 1))
                print("----- Train loss: {} -----".format(round(tr_loss, 3)))

                if args['task'] == 'QA':
                    train_exact_match = round(100 * (correct_answers / nb_tr_examples), 3)
                    train_f1 = round(100 * (batch_f1 / nb_tr_examples), 3)
                    print("----- Train {} exact-match: {} % -----".format(args['task'], train_exact_match))
                    print("----- Train {} F1: {} % -----".format(args['task'], train_f1))

                elif args['task'] == 'Sbj_Classification':
                    print("----- Train Sbj acc: {} % -----".format(batch_accs_sbj[-1]))
                    print("----- Train Sbj F1: {} % -----".format(batch_f1s_sbj[-1]))

                elif args['task'] == 'Domain_Classification':
                    print("----- Train Domain acc: {} % -----".format(batch_accs_domain[-1]))
                    print("----- Train Domain F1: {} % -----".format(batch_f1s_domain[-1]))

                print("----------------------------------")
                print()
        
                if args['n_evals'] == 'one_per_epoch':
                    if task == 'QA' and args['evaluation_strategy'] == 'no_aux_targets':
                        # save model's current weights for aux targets (we evaluate model without information about aux targets)
                        current_aux_weights = model.qa_head.fc_qa.weight[:, 768:]
                            
                    val_losses, val_accs, val_f1s, model = val(
                                                               model=model,
                                                               tokenizer=tokenizer,
                                                               val_dl=val_dl,
                                                               args=args,
                                                               current_step=step,
                                                               epoch=epoch,
                                                               batch_size=batch_size,
                                                               val_losses=val_losses,
                                                               val_accs=val_accs,
                                                               val_f1s=val_f1s,
                                                               loss_func=loss_func,
                                                               sequential_transfer=True,
                                                               evaluation_strategy=args['evaluation_strategy'],
                                                               )

                    if task == 'QA' and args['evaluation_strategy'] == 'no_aux_targets':
                        with torch.no_grad():
                            model.qa_head.fc_qa.in_features += add_features
                            model.qa_head.fc_qa.weight = nn.Parameter(torch.cat((model.qa_head.fc_qa.weight, current_aux_weights.to(device)), 1))

                    # we want to store train exact-match accuracies and F1 scores for each task
                    # as often as we evaluate model on validation set
                    running_tasks = tasks[:]
                    
                    # after evaluation on dev set, move model back to train mode
                    model.train()

                    if epoch > 0 and early_stopping:
                        if val_losses[-1] > val_losses[-2] or epoch >= args['n_epochs'] - 2:
                            print("------------------------------------------")
                            print("----- Stopping training after {} steps -----".format(nb_tr_steps + len(train_dl) * epoch))
                            print("------------------------------------------")
                            print()

                            val_losses_all_tasks.append(val_losses)
                            val_accs_all_tasks.append(val_accs)
                            val_f1s_all_tasks.append(val_f1s)

                        if args['training_regime'] == 'soft_targets' and (task == 'Sbj_Class' or task == 'Domain_Class'):
                            model.eval()
                            eval_round = True
                            seq_pair = '(q, a)' if args['batch_presentation'] == 'alternating' and task == 'Sbj_Class' else '(q, c)'
                            print("----------------------------------------------------------------------------------------------------------------")
                            print("----- Performing an extra evaluation epoch to store model's output logits for each {} input sequence -------".format(seq_pair))
                            print("----------------------------------------------------------------------------------------------------------------")
                            print()
                        else:
                            break
                else:
                    if stop_training or epoch >= args['n_epochs'] - 2:
                        print("------------------------------------------")
                        print("----- Stopping training after {} steps -----".format(nb_tr_steps + len(train_dl) * epoch))
                        print("------------------------------------------")
                        print()

                        val_losses_all_tasks.append(val_losses)
                        val_accs_all_tasks.append(val_accs)
                        val_f1s_all_tasks.append(val_f1s)

                        if args['training_regime'] == 'soft_targets' and (task == 'Sbj_Class' or task == 'Domain_Class'):
                            model.eval()
                            eval_round = True
                            seq_pair = '(q, a)' if args['batch_presentation'] == 'alternating' and task == 'Sbj_Class' else '(q, c)'
                            print("----------------------------------------------------------------------------------------------------------------")
                            print("----- Performing an extra evaluation epoch to store model's output logits for each {} input sequence -------".format(seq_pair))
                            print("----------------------------------------------------------------------------------------------------------------")
                            print()
                        else:
                            break
            else:
                print("---------------------------------------------------------")
                print("----- Evaluation epoch finished. Back to training. ------")
                print("---------------------------------------------------------")
                print()
                break

    # return model in eval mode
    model.eval()
    return batch_losses, batch_accs_qa, batch_f1s_qa, batch_accs_sbj, batch_f1s_sbj, batch_accs_domain, batch_f1s_domain, val_losses_all_tasks, val_accs_all_tasks, val_f1s_all_tasks, model