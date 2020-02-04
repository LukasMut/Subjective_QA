__all__ = [
           'freeze_bottom_n_layers',
           '_get_answers',
           '_to_cpu',
          ]

import torch
import transformers
import re 

def _get_answers(
                b_input_ids:torch.Tensor,
                start_logits:torch.Tensor,
                end_logits:torch.Tensor,
                ):
    """
    Args:
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

def _to_cpu(
            tensor:torch.Tensor,
            to_numpy:bool=False,
            ):
    """
    Args:
        tensor (torch.Tensor): tensor to be casted onto CPU
        to_numpy (bool): whether PyTorch tensor should be converted into NumPy array
    Return:
        tensor (np.ndarray OR torch.Tensor)
    """
    tensor = tensor.detach().cpu()
    if to_numpy:
        return tensor.numpy()
    else:
        return tensor

def freeze_bottom_n_layers(
                           model,
                           n:int,
                           model_name:str,
                           ):
    """freeze bottom N transformer layers / attention heads of model
    Args:
        model (pretrained BERT or RoBERTa transformer model)
        n (int): number of layers we want to freeze
        model_name (str): name of the pretrained model (must be one of {roberta, bert, distilbert})
    Return:
        model whose first N bottom layers are frozen (i.e., weights won't be updated during backpropagation)
    """
    model_names = ['roberta', 'bert', 'distilbert']
    model_name = model_name.lower()
    if model_name not in model_names:
        raise ValueError('Wrong model name provided. Model name must be one of {roberta, bert, distilbert}')
        
    for name, param in model.named_parameters():
        if name.startswith(model_name):
            layer_to_freeze = '.transformer.layer.' if model_name == 'distilbert' else '.encoder.layer.'
            if re.search(r'^' + model_name + layer_to_freeze, name):
                name = name.lstrip(model_name + layer_to_freeze)
            try:
                n_layer = int(name[:2])
                if n_layer < n:
                    param.requires_grad = False
            except:
                try:
                    n_layer = int(name[0])
                    if n_layer < n:
                        param.requires_grad = False
                except:
                    if name.startswith(model_name + '.embeddings'):
                        param.requires_grad = False
    return model