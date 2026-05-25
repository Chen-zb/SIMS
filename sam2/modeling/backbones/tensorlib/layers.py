#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import math
from typing import Optional, List

class TDLayer():
    def __init__(
        self,
        type,
        config,
        dropout
    ):
        self.type = type
        self.task_num = config['task_num']
        self.config = config
        self.td_dropout = dropout
        
        if dropout > 0.:
            self.td_dropout = nn.Dropout(dropout)
        else:
            self.td_dropout = lambda x: x


def shared_lora(num_embeddings, embedding_dim, config):
    A = nn.Parameter(torch.zeros([config['R'], num_embeddings], requires_grad=True))
    B = nn.Parameter(torch.zeros([embedding_dim, config['R']], requires_grad=True))
    nn.init.normal_(A)
    nn.init.zeros_(B)
    return A, B


def parafac2(config):
    pass


class QKVLinear(nn.Linear, TDLayer):
    def __init__(
        self, 
        in_features: int, 
        out_features: int,
        enable_qkv: List[bool],
        config,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        TDLayer.__init__(self, type=config['type'], config=config, dropout=config['dropout_rate'])
        assert out_features % len(enable_qkv) == 0, \
            'The length of enable_qkv must divide out_features'
        self.enable_qkv = enable_qkv
        self.task_num = config['task_num']
        self.scaling = config['scaling']
        
        # Actual trainable parameters
        if any(enable_qkv):
            
            if self.type == 'shared_lora':
                self.lora_A, self.lora_B = shared_lora(in_features, out_features // len(enable_qkv) * sum(enable_qkv), config)
            self.weight.requires_grad = False
            # Compute the indices
            self.ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_qkv), -1)
            self.ind[enable_qkv, :] = True
            self.ind = self.ind.view(-1)
    
    def generate_tensor(self, t=None):
        if self.type == 'shared_lora':
            tensor = self.lora_B.new_zeros([self.task_num, self.lora_B.shape[0], self.lora_A.shape[1]])
            for _ in range(self.task_num):
                tensor[_] = torch.einsum('mr,rn->mn', self.lora_B, self.lora_A)
            
        return tensor * self.scaling

    def zero_pad(self, x):
        result = x.new_zeros(*x.shape[:-2],len(self.ind), *x.shape[-1:])
        result[:, self.ind, :] = x
        return result

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)     

    def forward(self, x: torch.Tensor, task_idx=[-1], terra_t=None):
        tensor = self.generate_tensor(terra_t)
        tensor = self.zero_pad(tensor)
        result = F.linear(x, self.weight, bias=self.bias)
        if len(task_idx) == 1:
            result = F.linear(self.td_dropout(x), tensor[task_idx[0]])
        else:
            b, h, w, c = x.shape
            x = x.reshape(len(task_idx), -1, h, w, c)
            after_A = torch.concat([F.linear(self.td_dropout(x[idx]), tensor[_])
                                    for idx, _ in enumerate(task_idx)]
                                , dim=0)
            
            result=result + after_A
        
        return result

