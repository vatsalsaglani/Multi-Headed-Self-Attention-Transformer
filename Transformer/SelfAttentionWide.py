import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random, math 

from .utilities import mask_

torch.manual_seed(5)
    
class SelfAttentionWide(nn.Module):

    def __init__(self, emb, heads = 8, mask = False):

        super().__init__()

        self.emb = emb
        self.heads = heads
        self.mask = mask 

        self.tokeys = nn.Linear(emb, emb * heads, bias = False)
        self.toqueries = nn.Linear(emb, emb * heads, bias = False)
        self.tovalues = nn.Linear(emb, emb * heads, bias = False)

        self.unifyheads = nn.Linear(heads * emb, emb)


    def forward(self, x):

        b, t, e = x.size()
        h = self.heads 
        assert e == self.emb, f'Input embedding dimension {{e}} should match layer embedding dim {{self.emb}}'

        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)

        # dot-product attention

        # folding heads to batch dimensions

        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1/4))
        keys = keys / (e ** (1/4))



        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b*h, t, t)

        if self.mask:
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        # row wise self attention probabilities
        dot = F.softmax(dot, dim = 2)

        out = torch.bmm(dot, values).view(b, h, t, e)

        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        return self.unifyheads(out)

        




