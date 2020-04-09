import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import random, math 


from .utilities import mask_
    

torch.manual_seed(5)

class SelfAttentionNarrow(nn.Module):


    def __init__(self, emb, heads = 8, mask = False):

        super().__init__()

        assert emb % heads == 0, f'Embedding dimension {{emb}} should be divisible by number of heads {{heads}}'

        self.emb = emb 
        self.heads = heads 
        self.mask = mask 

        s = emb // heads

        self.tokeys = nn.Linear(s, s, bias=False)
        self.tovalues = nn.Linear(s, s, bias = False)
        self.toqueries = nn.Linear(s, s, bias=False)

        self.unifyheads = nn.Linear(heads * s, emb)


    def forward(self, x):

        b, t, e = x.size()

        h = self.heads
        assert e == self.emb, f'Input embedding dimension {{e}} should match layer embedding dimension {{self.emb}}'

        s = e // h
        x = x.view(b, t, h, s)

        keys = self.tokeys(x)
        queries = self.toqueries(x)
        values = self.tovalues(x)

        assert keys.size() == (b, t, h, s)
        assert queries.size() == (b, t, h, s)
        assert values.size() == (b, t, h, s)

        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        queries = queries / (e ** (1/4))
        keys = keys/(e ** (1/4))

        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b*h, t, t)

        if self.mask:
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim = 2)

        out = torch.bmm(dot, values).view(b, h, t, s)

        out = out.transpose(1, 2).contiguous().view(b, t, s * h)

        return self.unifyheads(out)

        