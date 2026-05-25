from distutils.command.config import config

import torch
import torch.nn as nn

def mark_only_td_as_trainable(model: nn.Module, if_norm=False, if_fix_core=False):
    for n, p in model.named_parameters():
        p.requires_grad = False
        
        if 'td_' in n:
            p.requires_grad = True
        
        if 'terra_' in n:
            p.requires_grad = True

        if 'lora_' in n:
            p.requires_grad = True
        
        if 'no_mask_embed' in n:
            p.requires_grad = True

        if 'convs.' in n:
            p.requires_grad = True

        if 'sam_mask_decoder' in n:
            p.requires_grad = True
            
        if 'norm' in n:
            p.requires_grad = True




def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )