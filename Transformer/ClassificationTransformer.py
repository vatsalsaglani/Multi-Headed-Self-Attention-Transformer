import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from .TransformerBlock import TransformerBlock


class ClassificationTransformer(nn.Module):

    def __init__(self, emb, heads, depth, seq_length, num_tokens, num_classes, device, ff_hidden_mult, max_pool = True, dropout = 0.0, wide = False, mask = False):

        super().__init__()
        
        self.num_tokens, self.max_pool, self.device, self.ff_hidden_mult = num_tokens, max_pool, device, ff_hidden_mult
        
        self.token_embedding = nn.Embedding(embedding_dim = emb, num_embeddings = num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim = emb, num_embeddings = seq_length)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(
                    emb = emb, heads = heads, mask = mask, seq_length=seq_length, ff_hidden_mult=self.ff_hidden_mult, dropout=dropout, wide = wide
                )
            )

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, num_classes)

        self.do = nn.Dropout(dropout)


    def forward(self, x):

        tokens = self.token_embedding(x)
        b, t, e = tokens.size()

        positions = self.pos_embedding(torch.arange(t, device = self.device))[None, :, :].expand(b, t, e)

        x = tokens + positions 
        x = self.do(x)

        x = self.tblocks(x)

        x = x.max(dim = 1)[0] if self.max_pool else x.mean(dim=1)

        x = self.toprobs(x)

        return F.log_softmax(x, dim = 1)


