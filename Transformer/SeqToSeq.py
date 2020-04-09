import torch 
import torch.nn as nn
import torch.nn.functional as F 

from .TransformerBlock import TransformerBlock
from .utilities import device_ as d 


torch.manual_seed(5)

class SequnceToSequence(nn.Module):
    """
    Transformer for generating text (character by character).
    """

    def __init__(self, emb, heads, depth, seq_length, num_tokens, num_op_tokens, device, mask = True, wide=False):
        super().__init__()

        self.num_op_tokens = num_op_tokens
        self.device = device
        self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)

        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=mask, wide=wide))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, num_op_tokens)

    def forward(self, x):
        """
        :param x: A (batch, sequence length) integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        tokens = self.token_embedding(x)
        b, t, e = tokens.size()

        positions = self.pos_embedding(torch.arange(t, device = self.device))[None, :, :].expand(b, t, e)
        x = tokens + positions

        x = self.tblocks(x)

        x = self.toprobs(x.view(b*t, e)).view(b, self.num_op_tokens, t)

        return F.log_softmax(x, dim=2)
