import torch 
import torch.nn as nn
import torch.nn.functional as F 
import math

from .TransformerBlock import TransformerBlock
from .utilities import device_ as d 


torch.manual_seed(5)

class SequnceToSequence(nn.Module):
    """
    Transformer for generating text (character by character).
    """

    def __init__(self, emb, heads, depth, seq_length, num_tokens, num_op_tokens, device, sinemb = False, mask = True, wide=False):
        super().__init__()

        self.num_op_tokens = num_op_tokens
        self.device = device
        self.seq_leb = seq_length
        self.sinemb = sinemb
        self.emb = emb
        self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)

        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=mask, wide=wide))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, num_op_tokens)


    @staticmethod
    def SinEmbedding(num_embeddings, embedding_dim):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype = torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype = torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim =1).view(num_embeddings, -1).unsqueeze(0)
        return emb



    def forward(self, x):
        """
        :param x: A (batch, sequence length) integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        tokens = self.token_embedding(x)
        b, t, e = tokens.size()

        if self.sinemb:
            positions = SequnceToSequence.SinEmbedding(t, e)
            positions = positions.to(d())
        else:
            positions = self.pos_embedding(torch.arange(t, device = self.device))[None, :, :].expand(b, t, e)

        x = tokens + positions

        x = self.tblocks(x)

        x = self.toprobs(x.view(b*t, e)).view(b, self.num_op_tokens, t)

        return F.log_softmax(x, dim=2)
