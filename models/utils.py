__all__ = [
           'accuracy',
           'f1',
           'freeze_transformer_layers',
           'get_answers',
           'compute_exact_batch',
           'compute_f1_batch',
           'to_cpu',
           'train',
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
from transformers import BertTokenizer, BertModel, BertForQuestionAnswering

from eval_squad import compute_exact, compute_f1

# set random seeds to reproduce results
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == "cuda":
  # set cuda random seeds 
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

def train(
          model,
          tokenizer,
          train_dl,
          val_dl,
          batch_size:int,
          args:dict,
          optimizer,
          scheduler=None,
          early_stopping:bool=True,
          n_aux_tasks=None,
          qa_type_weights=None,
          domain_weights=None,
          max_epochs:int=3,
          adversarial_simple:bool=False,
          plot_task_distrib:bool=False,
):
    n_iters = args['n_steps'] * args['n_epochs']
    n_examples = args['n_steps'] * batch_size
    
    if args['n_evals'] == 'multiple_per_epoch':
      steps_until_eval =  args['n_steps'] // args['n_evals_per_epoch'] # number of steps between validations
      stop_training = False
    
    L = 6 # total number of transformer layers in pre-trained DistilBERT model (L = 12 for BERT base, L = 6 for DistilBERT base)
    
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
          assert isinstance(qa_type_weights, torch.Tensor), 'Tensor of class weights for question-answer types is not provided'
          print("Weights for subjective Anwers: {}".format(qa_type_weights[0]))
          print()
          print("Weights for subjective Questions: {}".format(qa_type_weights[1]))
          print()

          # TODO: figure out, whether we need pos_weights for adversarial setting
          # loss func for auxiliary task to inform model about subjectivity (binary classification)
          
          sbj_loss_func = nn.BCEWithLogitsLoss(pos_weight=qa_type_weights.to(device))
          batch_accs_sbj, batch_f1s_sbj = [], []
        
          if n_aux_tasks == 2:
              assert isinstance(domain_weights, torch.Tensor), 'Tensor of class weights for different domains is not provided'
              # loss func for auxiliary task to inform model about different review / context domains (multi-way classification)
              domain_loss_func = nn.CrossEntropyLoss(weight=domain_weights.to(device))
              batch_accs_domain, batch_f1s_domain = [], []
              tasks.append('Domain_Class')

    elif args['task'] == 'Sbj_Classification':
      tasks = ['Sbj_Class']
      assert isinstance(qa_type_weights, torch.Tensor), 'Tensor of class weights for question-answer types is not provided'
      print("Weights for subjective Anwers: {}".format(qa_type_weights[0]))
      print()
      print("Weights for subjective Questions: {}".format(qa_type_weights[1]))
      print()
      sbj_loss_func = nn.BCEWithLogitsLoss(pos_weight=qa_type_weights.to(device))
      batch_accs_sbj, batch_f1s_sbj = [], []

    # generate random sample over all tasks (TODO: for MTL setting with 2 auxiliary tasks, we might want to sample QA task with a higher probability than auxiliary tasks)
    if isinstance(n_aux_tasks, type(None)) or args['task_sampling'] == 'uniform' or args['task'] == 'Sbj_Classification':
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

        if isinstance(n_aux_tasks, int):
          batch_acc_sbj, batch_f1_sbj = 0, 0

          if n_aux_tasks == 2:
            batch_acc_domain, batch_f1_domain = 0, 0

        if args['task'] == 'QA':
          correct_answers, batch_f1 = 0, 0

        elif args['task'] == 'Sbj_Classification':
          batch_acc_sbj, batch_f1_sbj = 0, 0

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        
        # n_steps == n_updates per epoch (n_iters = n_epochs * n_steps per epoch)
        for step, batch in enumerate(tqdm(train_dl, desc="Step")):

            if args['batch_presentation'] == 'alternating' and isinstance(n_aux_tasks, int):
              assert len(batch) == 2, 'In MTL, we must provide batches with different input sequences for the main and auxiliary task respectively'
              main_batch = tuple(t.to(device) for t in batch[0])
              aux_sbj_batch = tuple(t.to(device) for t in batch[1])

            else:
              main_batch = tuple(t.to(device) for t in batch)
            
            # sample task from random distribution
            current_task = task_order[step]

            batch_loss = 0            

            # zero-out gradients
            optimizer.zero_grad()
            
            if isinstance(n_aux_tasks, int):
              print('------------------------------------')
              print('-------- Current task: {} --------'.format(current_task))
              print('------------------------------------')
              print()

            if current_task == 'QA':

              # unpack inputs from dataloader for main task           
              b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, b_start_pos, b_end_pos, _, _ = main_batch

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

                if args['batch_presentation'] == 'alternating':
                  b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, b_sbj = main_batch
                else:
                  b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, _, _, b_sbj, _ = main_batch


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

                b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, _, _, _, b_domains = main_batch

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
              
              # keep track of steps taken per task (don't use overall steps)
              nb_tr_steps_aux = Counter(task_order[:step+1])[current_task]
              current_batch_acc_aux = round(100 * (batch_acc_aux / nb_tr_steps_aux), 3)
              current_batch_f1_aux = round(100 * (batch_f1_aux / nb_tr_steps_aux), 3)

              print("--------------------------------------------")
              print("----- Current batch {} acc: {} % -----".format(current_task, current_batch_acc_aux))
              print("----- Current batch {} F1: {} % -----".format(current_task, current_batch_f1_aux))
              print("--------------------------------------------")
              print()

              nb_tr_examples += b_input_ids.size(0)
              nb_tr_steps += 1

              # we don't want to save F1 scores and exact-match accuracies at the very beginning of training
              if step > (steps_until_eval // 2):

                if current_task in running_tasks:

                  if current_task == 'Sbj_Class':
                    batch_accs_sbj.append(current_batch_acc_aux)
                    batch_f1s_sbj.append(current_batch_f1_aux)

                  elif current_task == 'Domain_Class':
                    batch_accs_domain.append(current_batch_acc_aux)
                    batch_f1s_domain.append(current_batch_f1_aux)

                  running_tasks.pop(running_tasks.index(current_task))

            print("------------------------------------")
            print("----- Current {} loss: {} -----".format(current_task, abs(round(batch_loss.item(), 3))))
            print("------------------------------------")
            print()

            # In MTL setting, we just want to store QA losses (there's no need to store losses for auxiliary tasks since we want to observe effect on main task)
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

            # take step down the valley
            optimizer.step()
            
            # scheduler is only necessary, if we optimize with AdamW (BERT specific version of Adam with weight decay fix)
            if args['optim'] == 'AdamW' and not isinstance(scheduler, type(None)):
                scheduler.step()

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
                                                          loss_func=sbj_loss_func if args['task'] == 'Sbj_Classification' else qa_loss_func,
                                                          )

                # we want to store train exact-match accuracies and F1 scores for each task as often as we evaluate model on validation set
                running_tasks = tasks[:]
                  
                # after evaluation on dev set, move model back to train mode
                model.train()

                # want to train model at least for half an epoch
                if early_stopping and epoch > 0:
                  if val_losses[-1] > val_losses[-2] and val_losses[-1] > val_losses[-3]:
                    stop_training = True
                    break

        tr_loss /= task_distrib['QA'] if args['task'] == 'QA' else task_distrib['Sbj_Class']

        print("------------------------------------")
        print("---------- EPOCH {} ----------".format(epoch + 1))
        print("----- Train loss: {} -----".format(round(tr_loss, 3)))

        if args['task'] == 'QA':
          train_exact_match = round(100 * (correct_answers / (task_distrib['QA'] * batch_size)), 3)
          train_f1 = round(100 * (batch_f1 / (task_distrib['QA'] * batch_size)), 3)
          print("----- Train QA exact-match: {} % -----".format(round(val_exact_match, 3)))
          print("----- Train QA F1: {} % -----".format(round(val_f1, 3)))

          if isinstance(n_aux_tasks, int):

             print("------------------------------------")
             print("----- Train sbj acc: {} % -----".format(batch_accs_sbj[-1]))
             print("----- Train sbj F1: {} % -----".format(batch_f1s_sbj[-1]))
             print("------------------------------------")
             print()

             if n_aux_tasks == 2:

                print("------------------------------------")
                print("----- Train domain acc: {} % -----".format(batch_accs_domain[-1]))
                print("----- Train domain F1: {} % -----".format(batch_f1s_domain[-1]))
                print("------------------------------------")
                print()

        elif args['task'] == 'Sbj_Classification':
          print("----- Train Sbj acc: {} % -----".format(batch_accs_sbj[-1]))
          print("----- Train Sbj F1: {} % -----".format(batch_f1s_sbj[-1]))

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
                                                    loss_func=sbj_loss_func if args['task'] == 'Sbj_Classification' else qa_loss_func,
                                                    )

          # we want to store train exact-match accuracies and F1 scores for each task as often as we evaluate model on validation set
          running_tasks = tasks[:]

          # after evaluation on dev set, move model back to train mode
          model.train()

          if early_stopping and epoch > 0:
            if args['n_evals'] == 'one_per_epoch':
              if val_losses[-1] > val_losses[-2]:
                print("------------------------------------------")
                print("----- Early stopping after {} steps -----".format(nb_tr_steps * (epoch + 1)))
                print("------------------------------------------")
                break
        else:
          if stop_training:
            print("------------------------------------------")
            print("----- Early stopping after {} steps -----".format(nb_tr_steps * (epoch + 1)))
            print("------------------------------------------")
            break

    # return model in eval mode
    model.eval()
    if isinstance(n_aux_tasks, type(None)) and args['task'] == 'QA':
      return batch_losses, batch_accs_qa, batch_f1s_qa, val_losses, val_accs, val_f1s, model
    elif isinstance(n_aux_tasks, type(None)) and args['task'] == 'Sbj_Classification':
      return batch_losses, batch_accs_sbj, batch_f1s_sbj, val_losses, val_accs, val_f1s, model
    elif n_aux_tasks == 1:
      return batch_losses, batch_accs_qa, batch_f1s_qa, batch_accs_sbj, batch_f1s_sbj, val_losses, val_accs, val_f1s, model
    elif n_aux_tasks == 2:
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
):
    ### Validation ###

    # set model to eval mode
    model.eval()

    # path to save models
    model_path = args['model_dir'] 
    
    if args['task'] == 'QA':
      correct_answers_val = 0
    elif args['task'] == 'Sbj_Classification':
      batch_acc_sbj = 0
    
    batch_f1_val = 0
    val_loss = 0
    nb_val_steps, nb_val_examples = 0, 0

    for batch in val_dl:
        
        batch_loss_val = 0
        
        # add batch to current device
        batch = tuple(t.to(device) for t in batch)

        if args['task'] == 'Sbj_Classification':
          if  args['batch_presentation'] == 'alternating':
            b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, b_sbj = batch
          else:
            b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, _, _, b_sbj, _ = batch
        elif args['task'] == 'QA':
            b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, b_start_pos, b_end_pos,  _, _ = batch


         # if current batch_size is smaller than specified batch_size, skip batch
        if b_input_ids.size(0) != batch_size:
            continue
        
        # we evaluate the model on the main task only (i.e., QA)
        with torch.no_grad():

          if args['task'] == 'QA':

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
            
            current_batch_f1 = 100 * (batch_f1_val / nb_val_examples)
            current_batch_acc = 100 * (correct_answers_val / nb_val_examples)

          elif args['task'] == 'Sbj_Classification':

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

          print("----------------------------------------")
          print("----- Current val batch loss: {} -----".format(round(batch_loss_val.item(), 3)))
          print("----------------------------------------")
          print()

          val_loss += batch_loss_val.item()
          nb_val_examples += b_input_ids.size(0)
          nb_val_steps += 1

    val_loss /= nb_val_steps
    print("----------------------------------")
    print("-------- Train step {} --------".format(current_step + 1))
    print("----- Val loss: {} -----".format(round(val_loss, 3)))

    if args['task'] == 'QA':
      val_exact_match = 100 * (correct_answers_val / nb_val_examples)
      val_f1 = 100 * (batch_f1_val / nb_val_examples)

      print("----- Val QA exact-match: {} % -----".format(round(val_exact_match, 3)))
      print("----- Val QA F1: {} % -----".format(round(val_f1, 3)))

      if epoch == 0 or (val_exact_match > val_accs[-1] and val_f1 > val_f1s[-1]):
        torch.save(model.state_dict(), model_path + '/%s' % (args['model_name']))
    
    elif args['task'] == 'Sbj_Classification':
      val_acc = 100 * (batch_acc_sbj / nb_val_steps)
      val_f1 = 100 * (batch_f1_val / nb_val_steps)

      print("----- Val Sbj acc: {} % -----".format(round(val_acc, 3)))
      print("----- Val Sbj F1: {} % -----".format(round(val_f1, 3)))

      if epoch == 0 or (val_acc > val_accs[-1] and val_f1 > val_f1s[-1]):
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
        input_sequence:str='question_context',
):
    n_steps = len(test_dl)
    n_examples = n_steps * batch_size
       
    ### Inference ###

    # set model to eval mode
    model.eval()
    
    if task == 'QA':
      correct_answers_test = 0
      loss_func = nn.CrossEntropyLoss()

    elif args['task'] == 'Sbj_Classification':
      batch_acc_test = 0
      loss_func = nn.BCEWithLogitsLoss()
    
    batch_f1_test = 0
    test_f1, test_loss = 0, 0
    nb_test_steps, nb_test_examples = 0, 0

    for batch in test_dl:
       
        batch_loss_test = 0

        # move tensors in batch to current device (e.g., GPU)
        batch = tuple(t.to(device) for t in batch)
        
        if task == 'QA':
          b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, b_start_pos, b_end_pos, _, _ = batch

        elif task == 'Sbj_Classification':
          if input_sequence == 'question_context':
            b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, _, _, b_sbj, _ = batch
          elif input_sequence == 'question_answer':
            b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, b_sbj = batch

        # if current batch_size is smaller than specified batch_size, skip batch (number of examples in last batche might not equal to batch_size)
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
                  start_logits_test, end_logits_test = model(
                                                           input_ids=b_input_ids,
                                                           attention_masks=b_attn_masks,
                                                           token_type_ids=b_token_type_ids,
                                                           input_lengths=b_input_lengths,
                                                           task='QA',
                  )

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

              batch_acc_test += (current_sbj_acc / b_sbj.size(1))
              batch_f1_test += (current_sbj_f1 / b_sbj.size(1))

            test_loss += batch_loss_test.item()
            nb_test_examples += b_input_ids.size(0)
            nb_test_steps += 1
            
            current_batch_f1 = 100 * (batch_f1_test / nb_test_examples) if task == 'QA' else batch_f1_test / nb_test_steps 
            current_batch_acc = 100 * (correct_answers_test / nb_test_examples) if task == 'QA' else batch_acc_test / nb_test_steps 

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
    
    elif task == 'Sbj_Classification':

      test_acc = 100 * (batch_acc_test / nb_test_steps)
      test_f1 = 100 * (batch_f1_test / nb_test_steps)

      print("----- Test Sbj acc: {} % -----".format(round(val_acc, 3)))
      print("----- Test Sbj F1: {} % -----".format(round(val_f1, 3)))

    print("----------------------------------")
    print()
   
    return test_loss, test_exact_match, test_f1