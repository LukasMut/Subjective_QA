__all__ = [
           'freeze_transformer_layers',
           'sort_batch',
           'get_answers',
           'compute_exact_batch',
           'compute_f1_batch',
           'to_cpu',
           'train',
           'test',
]

import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F

import torch
import transformers

from collections import Counter, defaultdict
from tqdm import trange, tqdm
from transformers import BertTokenizer, BertModel, BertForQuestionAnswering

from eval_squad import compute_exact, compute_f1

# set random seeds to reproduce results
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
"""

def freeze_transformer_layers(
                              model,
                              model_name:str='bert',
                              unfreeze:bool=False,
):
    model_names = ['roberta', 'bert',]
    model_name = model_name.lower()
    if model_name not in model_names:
        raise ValueError('Incorrect model name provided. Model name must be one of {}'.format(model_names))
    for n, p in model.named_parameters():
        if n.startswith(model_name):
            p.requires_grad = True if unfreeze else False
    return model

# sort sequences in decreasing order w.r.t. to orig. sequence length
def sort_batch(
               input_ids:torch.Tensor,
               attn_masks:torch.Tensor,
               token_type_ids:torch.Tensor,
               input_lengths:torch.Tensor,
               start_pos:torch.Tensor,
               end_pos:torch.Tensor,
               q_sbj:torch.Tensor,
               a_sbj:torch.Tensor,
               domains:torch.Tensor,
               PAD_token:int=0,
):
    indices, input_ids = zip(*sorted(enumerate(to_cpu(input_ids)), key=lambda seq: len(seq[1][seq[1] != PAD_token]), reverse=True))
    indices = np.array(indices) if isinstance(indices, list) else np.array(list(indices))
    input_ids = torch.tensor(np.array(list(input_ids)), dtype=torch.long).to(device)
    return input_ids, attn_masks[indices], token_type_ids[indices], input_lengths[indices], start_pos[indices], end_pos[indices], q_sbj[indices], a_sbj[indices], domains[indices]

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
):
    n_iters = len(train_dl)
    n_examples = n_iters * batch_size
    max_epochs = 4
    
    if args["freeze_bert"]:
        model = freeze_transformer_layers(model)
        print("--------------------------------------------------")
        print("------ Pre-trained BERT weights are frozen -------")
        print("--------------------------------------------------")
        print()
        
    # store loss and accuracy for plotting
    batch_losses = []
    train_losses = []
    train_accs = []
    train_f1s = []
    val_losses = []
    val_accs = []
    val_f1s = []
    
    # path to save models
    model_path = args['model_dir'] 
    
    # define loss function (Cross-Entropy is numerically more stable than LogSoftmax plus Negative-Log-Likelihood)
    qa_loss_func = nn.CrossEntropyLoss()
    
    if isinstance(n_aux_tasks, int):
        
        # loss func for auxiliary task to inform model about subjectivity (binary classification)
        assert isinstance(qa_type_weights, torch.Tensor), 'Tensor of class weights for question-answer types is not provided'
        #assert len(qa_type_weights) == 1, 'For binary cross-entropy loss, we must provide a single weight for positive examples'
        print("Weights for subjective QAs: {}".format(qa_type_weights))
        print()
        
        if n_aux_tasks == 1:
            sbj_loss_func = nn.BCEWithLogitsLoss(pos_weight=qa_type_weights.to(device))
        
        # loss func for auxiliary task to inform model about different review / context domains (multi-way classification)
        elif n_aux_tasks == 2:
            sbj_loss_func = nn.BCEWithLogitsLoss(pos_weight=qa_type_weights.to(device))
            assert isinstance(domain_weights, torch.Tensor), 'Tensor of class weights for different domains is not provided'
            domain_loss_func = nn.CrossEntropyLoss(weight=domain_weights.to(device))
    
    if args['freeze_bert'] and (args['dataset'] == 'SubjQA' or args['dataset'] == 'combined'):
        if args['n_epochs'] <= 3:
            # add an additional epoch for fine-tuning (not only the heads but) the entire model (+ BERT encoder)
            args['n_epochs'] += (max_epochs - args['n_epochs'])

    for epoch in trange(args['n_epochs'],  desc="Epoch"):

        ### Training ###

        model.train()
        
        # if last training epoch
        if epoch == args['n_epochs'] - 1 and (args['dataset'] == 'SubjQA' or args['dataset'] == 'combined'):
            model = freeze_transformer_layers(model, unfreeze=True)
            print("------------------------------------------------------------")
            print("---------- Pre-trained BERT weights are unfrozen -----------")
            print("------------------------------------------------------------")
            print("------ Entire model will be trained for single epoch -------")
            print("------------------------------------------------------------")
            print()

        correct_answers, batch_f1 = 0, 0
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        
        # number of steps == number of updates per epoch
        for i, batch in enumerate(tqdm(train_dl, desc="Step")):
            
            batch_loss, qa_loss, sbj_loss, domain_loss = 0, 0, 0, 0 

            # add batch to GPU
            batch = tuple(t.to(device) for t in batch)

            # unpack inputs from dataloader            
            b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, b_start_pos, b_end_pos, b_cls_indexes, _, b_q_sbj, b_a_sbj, b_domains, _ = batch
            
            if args["sort_batch"]:
                # sort sequences in batch in decreasing order w.r.t. to (original) sequence length
                b_input_ids, b_attn_masks, b_type_ids, b_input_lengths, b_start_pos, b_end_pos, b_q_sbj, b_a_sbj, b_domains = sort_batch(
                                                                                                                        b_input_ids,
                                                                                                                        b_attn_masks,
                                                                                                                        b_token_type_ids,
                                                                                                                        b_input_lengths,
                                                                                                                        b_start_pos,
                                                                                                                        b_end_pos,
                                                                                                                        b_q_sbj,
                                                                                                                        b_a_sbj,
                                                                                                                        b_domains,
                )
            
            if args['optim'] == 'SGD' and not isinstance(scheduler, type(None)):
                scheduler.step(epoch + i / n_iters)
            
            # zero-out gradients
            optimizer.zero_grad()
            
            if isinstance(n_aux_tasks, type(None)):
                # compute start and end logits respectively
                start_logits, end_logits = model(
                                                 input_ids=b_input_ids,
                                                 attention_masks=b_attn_masks,
                                                 token_type_ids=b_token_type_ids,
                                                 input_lengths=b_input_lengths,
                )
                
            elif isinstance(n_aux_tasks, int):
                if n_aux_tasks == 1:
                    ans_logits, sbj_logits = model(
                                                   input_ids=b_input_ids,
                                                   attention_masks=b_attn_masks,
                                                   token_type_ids=b_token_type_ids,
                                                   input_lengths=b_input_lengths,
                )
                    start_logits, end_logits = ans_logits
                    
                    # compute auxiliary loss (subjectivity loss)
                    if args['qa_type'] == 'question':
                        b_q_sbj = b_q_sbj.type_as(sbj_logits)
                        sbj_loss = sbj_loss_func(sbj_logits, b_q_sbj)
                        
                    elif args['qa_type'] == 'answer':
                        b_a_sbj = b_a_sbj.type_as(sbj_logits)
                        sbj_loss = sbj_loss_func(sbj_logits, b_a_sbj)
                    
                elif n_aux_tasks == 2:
                    ans_logits, sbj_logits, domain_logits = model(
                                                                  input_ids=b_input_ids,
                                                                  attention_masks=b_attn_masks,
                                                                  token_type_ids=b_token_type_ids,
                                                                  input_lengths=b_input_lengths,
                )
                    start_logits, end_logits = ans_logits
                    
                    # compute auxiliary losses
                    if args['qa_type'] == 'question':
                        b_q_sbj = b_q_sbj.type_as(sbj_logits)
                        sbj_loss = sbj_loss_func(sbj_logits, b_q_sbj)
                        
                    elif args['qa_type'] == 'answer':
                        b_a_sbj = b_a_sbj.type_as(sbj_logits)
                        sbj_loss = sbj_loss_func(sbj_logits, b_a_sbj)
                    
                    domain_loss = domain_loss_func(domain_logits, b_domains)
                
                """
                elif n_aux_tasks == 3:
                    ans_logits, sbj_logits, domain_logits, ds_logits = model(
                                                                             input_ids=b_input_ids,
                                                                             attention_masks=b_attn_masks,
                                                                             token_type_ids=b_token_type_ids,
                                                                             input_lengths=b_input_lengths,
                )
                    start_logits, end_logits = ans_logits                    
                """
            
            # start and end loss must be computed separately
            start_loss = qa_loss_func(start_logits, b_start_pos)
            end_loss = qa_loss_func(end_logits, b_end_pos)
            qa_loss = (start_loss + end_loss) / 2
            
            # accumulate all losses
            if isinstance(n_aux_tasks, type(None)):
                batch_loss += qa_loss
                
            elif n_aux_tasks == 1:
                batch_loss += (qa_loss + sbj_loss) / 2
                
            elif n_aux_tasks == 2:
                batch_loss += (qa_loss + sbj_loss + domain_loss) / 3
            
            print("------------------------------------")
            print("----- Current batch loss: {} -----".format(round(batch_loss.item(), 3)))
            print("------------------------------------")
            print()

            batch_losses.append(batch_loss.item())
            
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
                        
            # backpropagate error
            
            #TODO: figure out how to backpropagte errors for MTL
            #qa_loss.backward()
            #sbj_loss.backward()
            #domain_loss.backward()
            
            batch_loss.backward()
            
            # clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), args["max_grad_norm"])

            # update model parameters and take a step using the computed gradient
            optimizer.step()
            
            # scheduler is only necessary, if we optimize through AdamW (BERT specific version of Adam)
            if args['optim'] == 'AdamW' and not isinstance(scheduler, type(None)):
                scheduler.step()

            tr_loss += batch_loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            
            current_batch_f1 = 100 * (batch_f1 / nb_tr_examples)
            current_batch_acc = 100 * (correct_answers / nb_tr_examples)
            
            print("--------------------------------------------")
            print("----- Current batch exact-match: {} % -----".format(round(current_batch_acc, 3)))
            print("----- Current batch F1: {} % -----".format(round(current_batch_f1, 3)))
            print("--------------------------------------------")
            print()
                    
        train_loss = tr_loss / nb_tr_steps
        train_exact_match = 100 * (correct_answers / nb_tr_examples)
        train_f1 = 100 * (batch_f1 / nb_tr_examples)
        
        print("-------------------------------")
        print("---------- EPOCH {} ----------".format(epoch))
        print("----- Train loss: {} -----".format(round(tr_loss / nb_tr_steps, 3)))
        print("----- Train exact-match: {} % -----".format(round(train_exact_match, 3)))
        print("----- Train F1: {} % -----".format(round(train_f1, 3)))
        print("-------------------------------")
        print()

        train_losses.append(train_loss)
        train_accs.append(train_exact_match)
        train_f1s.append(train_f1)
       
        ### Validation ###

        # set model to eval mode
        model.eval()
        
        correct_answers_val, batch_f1_val = 0, 0
        val_loss = 0
        nb_val_steps, nb_val_examples = 0, 0

        for batch in val_dl:
            
            batch_loss_val = 0
            
            # add batch to current device
            batch = tuple(t.to(device) for t in batch)

            # unpack inputs from dataloader            
            b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, b_start_pos, b_end_pos, b_cls_indexes, _, _, _, _, _ = batch
            
            if args["sort_batch"]:
                # sort sequences in batch in decreasing order w.r.t. to (original) sequence length
                b_input_ids, b_attn_masks, b_type_ids, b_input_lengths, b_start_pos, b_end_pos = sort_batch(
                                                                                                            b_input_ids,
                                                                                                            b_attn_masks,
                                                                                                            b_token_type_ids,
                                                                                                            b_input_lengths,
                                                                                                            b_start_pos,
                                                                                                            b_end_pos,
                )
            
            with torch.no_grad():
                
                # compute start and end logits respectively
                start_logits_val, end_logits_val = model(
                                                         input_ids=b_input_ids,
                                                         attention_masks=b_attn_masks,
                                                         token_type_ids=b_token_type_ids,
                                                         input_lengths=b_input_lengths,
                )

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

        val_loss = val_loss / nb_val_steps
        val_exact_match = 100 * (correct_answers_val / nb_val_examples)
        val_f1 = 100 * (batch_f1_val / nb_val_examples)
        
        print("----------------------------------")
        print("---------- EPOCH {} ----------".format(epoch))
        print("----- Val loss: {} -----".format(round(val_loss, 3)))
        print("----- Val exact-match: {} % -----".format(round(val_exact_match, 3)))
        print("----- Val F1: {} % -----".format(round(val_f1, 3)))
        print("----------------------------------")
        print()
        
        if (epoch == 0) or (val_exact_match > val_accs[-1]):
            torch.save(model.state_dict(), model_path + '/%s' % (args.model_name))
        
        val_losses.append(val_loss)
        val_accs.append(val_exact_match)
        val_f1s.append(val_f1)
        
        if epoch > 0 and early_stopping:
            if (val_accs[-2] > val_accs[-1]) and (val_f1s[-2] > val_f1s[-1]):
                print("------------------------------------------")
                print("----- Early stopping after {} epochs -----".format(epoch))
                print("------------------------------------------")
                break
       
    return batch_losses, train_losses, train_accs, train_f1s, val_losses, val_accs, val_f1s, model

def test(
          model,
          tokenizer,
          test_dl,
          batch_size:int,
          sort_batch:bool=False,
          not_finetuned:bool=False,
):
    n_iters = len(test_dl)
    n_examples = n_iters * batch_size
       
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
        b_input_ids, b_attn_masks, b_token_type_ids, b_input_lengths, b_start_pos, b_end_pos, b_cls_indexes, _, _, _, _, _ = batch
        
        # if current batch_size is smaller than specified batch_size, skip batch
        if b_input_ids.size(0) != batch_size:
            questions_not_tested = b_input_ids.size(0)
            continue

        if sort_batch:
            # sort sequences in batch in decreasing order w.r.t. to (original) sequence length
            b_input_ids, b_attn_masks, b_type_ids, b_input_lengths, b_start_pos, b_end_pos = sort_batch(
                                                                                                        b_input_ids,
                                                                                                        b_attn_masks,
                                                                                                        b_token_type_ids,
                                                                                                        b_input_lengths,
                                                                                                        b_start_pos,
                                                                                                        b_end_pos,
            )

        with torch.no_grad():
            
            if not_finetuned:
                start_logits_test, end_logits_test = model(
                                                           input_ids=b_input_ids,
                                                           attention_mask=b_attn_masks,
                                                           token_type_ids=b_token_type_ids,
                )

            else:  
                start_logits_test, end_logits_test = model(
                                                         input_ids=b_input_ids,
                                                         attention_masks=b_attn_masks,
                                                         token_type_ids=b_token_type_ids,
                                                         input_lengths=b_input_lengths,
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