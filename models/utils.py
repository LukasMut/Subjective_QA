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
    n_steps = len(train_dl)
    n_iters = n_steps * args['n_epochs']
    n_examples = n_steps * batch_size
    
    if args['n_evals'] == 'multiple_per_epoch':
      steps_until_eval = n_steps // args['n_evals_per_epoch'] # number of steps between validations
    
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
    batch_accs_qa = []
    batch_f1s_qa = []
    val_losses = []
    val_accs = []
    val_f1s = []
    
    # define loss function (Cross-Entropy is numerically more stable than LogSoftmax plus Negative-Log-Likelihood Loss)
    qa_loss_func = nn.CrossEntropyLoss()

    tasks = ['QA']

    if isinstance(n_aux_tasks, int):
        
        tasks.append('Sbj_Class')
        assert isinstance(qa_type_weights, torch.Tensor), 'Tensor of class weights for question-answer types is not provided'
        print("Weights for subjective Anwers: {}".format(qa_type_weights[0]))
        print()
        print("Weights for subjective Questions: {}".format(qa_type_weights[1]))
        print()

        # TODO: figure out, whether we need pos_weights for adversarial setting
        # loss func for auxiliary task to inform model about subjectivity (binary classification)
        
        # NOTE: pos_weight approach does not work really well (worse than BCE without higher weights for subjective questions)
        sbj_loss_func = nn.BCEWithLogitsLoss(pos_weight=qa_type_weights.to(device))
        #sbj_loss_func = nn.BCEWithLogitsLoss()
        batch_accs_sbj, batch_f1s_sbj = [], []
      
        if n_aux_tasks == 2:
            assert isinstance(domain_weights, torch.Tensor), 'Tensor of class weights for different domains is not provided'
            # loss func for auxiliary task to inform model about different review / context domains (multi-way classification)
            domain_loss_func = nn.CrossEntropyLoss(weight=domain_weights.to(device))
            batch_accs_domain, batch_f1s_domain = [], []
            tasks.append('Domain_Class')

    # generate uniform random sample over all entries (for MTL setting with 2 auxiliary tasks, we might want to sample QA task with a higher probability)
    task_order = np.random.choice(tasks, size=n_steps, replace=True, p = [1/len(tasks) for _ in tasks])
    task_distrib = Counter(task_order)

    if plot_task_distrib:
      plt.bar(tasks, [task_distrib[task] for task in tasks], alpha=0.5, edgecolor='black')
      plt.xticks(range(len(tasks)), labels=tasks)
      plt.xlabel('Tasks', fontsize=12)
      plt.ylabel('Frequency per epoch', fontsize=12)
      plt.title('Task distribution in MTL setting')
      plt.show()
      plt.clf()


    for epoch in trange(args['n_epochs'],  desc="Epoch"):

        ### Training ###

        model.train()
        
        """
        ## NOTE: for now, we don't use this training regime ##
        if args["freeze_bert"]:
          # gradually unfreeze layer by layer after the first epoch (no updating of BERT weights before task-specific layers haven't been trained)
          if epoch > 0 and (args['dataset'] == 'SubjQA' or args['dataset'] == 'combined'):
              model = freeze_transformer_layers(model, model_name=model_name, unfreeze=True, l=l)
              print("------------------------------------------------------------------------------------------")
              print("---------- Pre-trained BERT weights of top {} transformer layers are unfrozen -----------".format(L - l ))
              print("-------------------------------------------------------------------------------------------")
              print()
              l -= k
        """

        if isinstance(n_aux_tasks, int):
          batch_acc_sbj, batch_f1_sbj = 0, 0

          if n_aux_tasks == 2:
            batch_acc_domain, batch_f1_domain = 0, 0

        correct_answers, batch_f1 = 0, 0
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        
        # n_steps == n_updates per epoch (n_iters = n_epochs * n_steps per epoch)
        for i, batch in enumerate(tqdm(train_dl, desc="Step")):
            
            batch_loss = 0 

            # add batch to GPU
            batch = tuple(t.to(device) for t in batch)

            # unpack inputs from dataloader            
            b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, b_start_pos, b_end_pos, b_cls_indexes, _, b_sbj, b_domains, _ = batch
            
            optimizer.zero_grad()

            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            
            current_task = task_order[i]

            if isinstance(n_aux_tasks, int):
              print('------------------------------------')
              print('-------- Current task: {} --------'.format(current_task))
              print('------------------------------------')
              print()

            if current_task == 'QA':

              start_logits, end_logits = model(
                             input_ids=b_input_ids,
                             attention_masks=b_attn_masks,
                             token_type_ids=b_token_type_ids,
                             input_lengths=b_input_lengths,
                             task=current_task,
                             )

              # start and end loss must be computed separately
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
              nb_tr_examples_qa = Counter(task_order[:i+1])[current_task] * batch_size

              current_batch_acc = round(100 * (correct_answers / nb_tr_examples_qa), 3)
              current_batch_f1 = round(100 * (batch_f1 / nb_tr_examples_qa), 3)
              
              print("--------------------------------------------")
              print("----- Current batch {} exact-match: {} % -----".format(current_task, current_batch_acc))
              print("----- Current batch {} F1: {} % -----".format(current_task, current_batch_f1))
              print("--------------------------------------------")
              print()

              batch_accs_qa.append(current_batch_acc)
              batch_f1s_qa.append(current_batch_f1)

            else:

              if current_task == 'Sbj_Class':

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
              nb_tr_steps_aux = Counter(task_order[:i+1])[current_task]
              current_batch_acc_aux = round(100 * (batch_acc_aux / nb_tr_steps_aux), 3)
              current_batch_f1_aux = round(100 * (batch_f1_aux / nb_tr_steps_aux), 3)
              
              if current_task == 'Sbj_Class':
                batch_accs_sbj.append(current_batch_acc_aux)
                batch_f1s_sbj.append(current_batch_f1_aux)
              
              elif current_task == 'Domain_Class':
                batch_accs_domain.append(current_batch_acc_aux)
                batch_f1s_domain.append(current_batch_f1_aux)

              print("--------------------------------------------")
              print("----- Current batch {} acc: {} % -----".format(current_task, current_batch_acc_aux))
              print("----- Current batch {} F1: {} % -----".format(current_task, current_batch_f1_aux))
              print("--------------------------------------------")
              print()

            print("------------------------------------")
            print("----- Current {} loss: {} -----".format(current_task, abs(round(batch_loss.item(), 3))))
            print("------------------------------------")
            print()

            # we just want to store QA losses
            if current_task == 'QA':
              tr_loss += batch_loss.item()
              batch_losses.append(batch_loss.item())

            batch_loss.backward()
            
            # clip gradients if gradients become larger than specified norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), args["max_grad_norm"])

            # take step down the valley
            optimizer.step()
            
            # scheduler is only necessary, if we optimize with AdamW (BERT specific version of Adam)
            if args['optim'] == 'AdamW' and not isinstance(scheduler, type(None)):
                scheduler.step()

            if args['n_evals'] == 'multiple_per_epoch':
              if (i > 0 and i % steps_until_eval == 0):
                val_losses, val_accs, val_f1s, model = val(
                                                          model=model,
                                                          tokenizer=tokenizer,
                                                          val_dl=val_dl,
                                                          qa_loss_func=qa_loss_func,
                                                          args=args,
                                                          current_step=i,
                                                          epoch=epoch,
                                                          batch_size=batch_size,
                                                          val_losses=val_losses,
                                                          val_accs=val_accs,
                                                          val_f1s=val_f1s,
                                                          )
                # set model back to train mode
                model.train()

        tr_loss /= task_distrib['QA']
        train_exact_match = round(100 * (correct_answers / (task_distrib['QA'] * batch_size)), 3)
        train_f1 = round(100 * (batch_f1 / (task_distrib['QA'] * batch_size)), 3)

        print("------------------------------------")
        print("---------- EPOCH {} ----------".format(epoch + 1))
        print("----- Train loss: {} -----".format(round(tr_loss, 3)))
        print("----- Train exact-match: {} % -----".format(train_exact_match))
        print("----- Train F1: {} % -----".format(train_f1))
        print("------------------------------------")
        print()

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
        
        if args['n_evals'] == 'one_per_epoch':
          val_losses, val_accs, val_f1s, model = val(
                                                    model=model,
                                                    tokenizer=tokenizer,
                                                    val_dl=val_dl,
                                                    qa_loss_func=qa_loss_func,
                                                    args=args,
                                                    current_step=i,
                                                    epoch=epoch,
                                                    batch_size=batch_size,
                                                    val_losses=val_losses,
                                                    val_accs=val_accs,
                                                    val_f1s=val_f1s,
                                                    )
          model.train()

        if epoch > 0 and early_stopping:
          if args['n_evals'] == 'one_per_epoch':
            if val_f1s[-1] < val_f1s[-2] and val_accs[-1] < val_accs[-2]:
                print("------------------------------------------")
                print("----- Early stopping after {} steps -----".format(nb_tr_steps * (epoch + 1)))
                print("------------------------------------------")
                break

          elif args['n_evals'] == 'multiple_per_epoch':      
            if (val_f1s[-1] < val_f1s[-2] and val_f1s[-1] < val_f1s[-3]) and (val_accs[-1] < val_accs[-2] and val_accs[-1] < val_accs[-3]):
                print("------------------------------------------")
                print("----- Early stopping after {} steps -----".format(nb_tr_steps * (epoch + 1)))
                print("------------------------------------------")
                break

    # return model in eval mode
    model.eval()
    if isinstance(n_aux_tasks, type(None)):
      return batch_losses, batch_accs_qa, batch_f1s_qa, val_losses, val_accs, val_f1s, model
    elif n_aux_tasks == 1:
      return batch_losses, batch_accs_qa, batch_f1s_qa, batch_accs_sbj, batch_f1s_sbj, val_losses, val_accs, val_f1s, model
    elif n_aux_tasks == 2:
      return batch_losses, batch_accs_qa, batch_f1s_qa, batch_accs_sbj, batch_f1s_sbj, batch_accs_domain, batch_f1s_domain, val_losses, val_accs, val_f1s, model

def val(
        model,
        tokenizer,
        val_dl,
        qa_loss_func,
        args:dict,
        current_step:int,
        epoch:int,
        batch_size:int,
        val_losses:list,
        val_accs:list,
        val_f1s:list,
):
    ### Validation ###

    # set model to eval mode
    model.eval()

    # path to save models
    model_path = args['model_dir'] 
    
    correct_answers_val, batch_f1_val = 0, 0
    val_loss = 0
    nb_val_steps, nb_val_examples = 0, 0

    for batch in val_dl:
        
        batch_loss_val = 0
        
        # add batch to current device
        batch = tuple(t.to(device) for t in batch)

        # unpack inputs from dataloader            
        b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, b_start_pos, b_end_pos, b_cls_indexes, _, _, _, _ = batch

         # if current batch_size is smaller than specified batch_size, skip batch
        if b_input_ids.size(0) != batch_size:
            continue
        
        # we evaluate the model on the main task only (i.e., QA)
        with torch.no_grad():

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
            start_loss = qa_loss_func(start_logits_val, b_start_pos)
            end_loss = qa_loss_func(end_logits_val, b_end_pos)
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
            
            val_loss += batch_loss_val.item()
            nb_val_examples += b_input_ids.size(0)
            nb_val_steps += 1
            
            current_batch_f1 = 100 * (batch_f1_val / nb_val_examples)
            current_batch_acc = 100 * (correct_answers_val / nb_val_examples)

    val_loss /= nb_val_steps
    val_exact_match = 100 * (correct_answers_val / nb_val_examples)
    val_f1 = 100 * (batch_f1_val / nb_val_examples)
    
    print("----------------------------------")
    print("-------- Train step {} --------".format(current_step + 1))
    print("----- Val loss: {} -----".format(round(val_loss, 3)))
    print("----- Val exact-match: {} % -----".format(round(val_exact_match, 3)))
    print("----- Val F1: {} % -----".format(round(val_f1, 3)))
    print("----------------------------------")
    print()

    if epoch == 0 or (val_exact_match > val_accs[-1] and val_f1 > val_f1s[-1]):
        torch.save(model.state_dict(), model_path + '/%s' % (args['model_name']))

    val_losses.append(val_loss)
    val_accs.append(val_exact_match)
    val_f1s.append(val_f1)

    return val_losses, val_accs, val_f1s, model

def test(
          model,
          tokenizer,
          test_dl,
          batch_size:int,
          not_finetuned:bool=False,
):
    n_steps = len(test_dl)
    n_examples = n_steps * batch_size
       
    ### Inference ###

    # set model to eval mode
    model.eval()
    
    # define loss function
    loss_func = nn.CrossEntropyLoss()

    correct_answers_test, batch_f1_test = 0, 0
    test_f1, test_loss = 0, 0
    nb_test_steps, nb_test_examples = 0, 0

    for batch in test_dl:
       
        batch_loss_test = 0

        # add batch to current device
        batch = tuple(t.to(device) for t in batch)

        # unpack inputs from dataloader            
        b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, b_start_pos, b_end_pos, _, _, _, _, _ = batch
        
        # if current batch_size is smaller than specified batch_size, skip batch
        if b_input_ids.size(0) != batch_size:
            continue

        with torch.no_grad():
            
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

            test_loss += batch_loss_test.item()
            nb_test_examples += b_input_ids.size(0)
            nb_test_steps += 1

            current_batch_f1 = 100 * (batch_f1_test / nb_test_examples)
            current_batch_acc = 100 * (correct_answers_test / nb_test_examples)
            
            print("--------------------------------------------")
            print("----- Current batch exact-match: {} % -----".format(round(current_batch_acc, 3)))
            print("----- Current batch F1: {} % -----".format(round(current_batch_f1, 3)))
            print("--------------------------------------------")
            print()

    test_loss = test_loss / nb_test_steps
    test_exact_match = 100 * (correct_answers_test / nb_test_examples)
    test_f1 = 100 * (batch_f1_test / nb_test_examples)
    
    print()
    print("------------------------------------")
    print("------------ Inference ------------")
    print("------- Test loss: {} -------".format(round(test_loss, 3)))
    print("----- Test exact-match: {} % -----".format(round(test_exact_match, 3)))
    print("------- Test F1: {} % -------".format(round(test_f1, 3)))
    print("------------------------------------")
    print()
   
    return test_loss, test_exact_match, test_f1