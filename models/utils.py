__all__ = [
           'get_answers',
           'to_cpu',
           'sort_batch',
           'freeze_transformer_layers',
]

import numpy as np

import torch
import transformers

# torch.cuda.is_available() checks and returns True if a GPU is available, else it'll return False
#is_cuda = torch.cuda.is_available()

#if is_cuda:
#    device = torch.device("cuda")
#    print("GPU is available")
#else:

device = torch.device("cpu")
print("GPU not available, CPU used")

def get_answers(
                tokenizer,
                b_input_ids:torch.Tensor,
                start_logits:torch.Tensor,
                end_logits:torch.Tensor,
):
    """
    Args:
        tokenizer (BERT tokenizer)
        b_input_ids (torch.Tensor): batch of inputs IDs (batch_size x 512)
        start_logits (torch.Tensor): model's output logits for start positions
        end_logits (torch.Tensor): model's output logits for end positions
    Return:
        answers (list): list of predicted answers (str)
    """
    answers = []
    for input_ids, start_log, end_log in zip(b_input_ids, start_logits, end_logits):
        all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        answer = ' '.join(all_tokens[torch.argmax(start_log):torch.argmax(end_log) + 1])
        answers.append(answer)
    return answers

# move tensor to CPU
def to_cpu(
           tensor:torch.Tensor,
           detach:bool=False,
           to_numpy:bool=True,
):
    """
    Args:
        tensor (torch.Tensor): tensor to be moved to CPU
        detach (bool): whether tensor should be detached from computation graph (i.e., requires_grad = False)
        to_numpy (bool): whether torch.Tensor should be converted into np.ndarray
    Return:
        tensor (np.ndarray OR torch.Tensor)
    """
    tensor = tensor.detach().cpu() if detach else tensor.cpu()
    if to_numpy: return tensor.numpy()
    else: return tensor

    
# sort sequences in decreasing order w.r.t. to orig. sequence length
def sort_batch(
               input_ids:torch.Tensor,
               attn_masks:torch.Tensor,
               token_type_ids:torch.Tensor,
               input_lengths:torch.Tensor,
               start_pos:torch.Tensor,
               end_pos:torch.Tensor,
               PAD_token:int=0,
):
    indices, input_ids = zip(*sorted(enumerate(to_cpu(input_ids)), key=lambda seq: len(seq[1][seq[1] != PAD_token]), reverse=True))
    indices = np.array(indices) if isinstance(indices, list) else np.array(list(indices))
    input_ids = torch.tensor(np.array(list(input_ids)), dtype=torch.long).to(device)
    return input_ids, attn_masks[indices], token_type_ids[indices], input_lengths[indices], start_pos[indices], end_pos[indices]

def freeze_transformer_layers(
                              model,
                              model_name:str='bert',
):
    """freeze transformer layers (necessary, if we want to train different QA heads on SQuAD)
    Args:
        model (pre-trained BERT transformer model)
        model_name (str): name of the pre-trained transformer model
    Return:
        QA model whose transformer layers are frozen (i.e., BERT weights won't be updated during backpropagation)
    """
    model_names = ['roberta', 'bert',]
    model_name = model_name.lower()
    if model_name not in model_names:
        raise ValueError('Incorrect model name provided. Model name must be one of {roberta, bert}')
        
    for n, p in model.named_parameters():
        if n.startswith(model_name):
            p.requires_grad = False
    return model