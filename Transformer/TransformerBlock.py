import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from .SelfAttentionNarrow import SelfAttentionNarrow
from .SelfAttentionWide import SelfAttentionWide

class TransformerBlock(nn.Module):

    def __init__(self, emb, heads, mask, seq_length, ff_hidden_mult = 4, dropout = 0.0, wide = True):

        super().__init__()

        self.attention = SelfAttentionWide(emb, heads = heads, mask = mask) if wide else SelfAttentionNarrow(emb, heads = heads, mask = mask)

        self.mask = mask 
        

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.do = nn.Dropout(dropout)


    def forward(self, x):

        attended = self.attention(x)

        x = self.norm1(attended + x)

        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do(x)

        return x