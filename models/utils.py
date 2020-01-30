__all__ = ['freeze_bottom_n_layers']

import transformers
import re 

def freeze_bottom_n_layers(model, n:int, model_name:str):
    """freeze bottom n transformer layers / attention heads of model
    Args:
        model (pretrained BERT or RoBERTa transformer model)
        n (int): number of layers we want to freeze
        model_name (str): name of the pretrained model (must be one of {roberta, bert, distilbert})
    Return:
        model whose first n bottom layers are frozen (i.e., weights won't be updated during backpropagation)
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