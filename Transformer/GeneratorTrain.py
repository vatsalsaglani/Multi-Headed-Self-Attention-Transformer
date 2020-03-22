import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.autograd import Variable


import math, random 
from tqdm import tqdm, trange


from .Generator import GeneratorTransformer
from .utilities import mask_
from .utilities import device_ as d 



def fit_generator(epochs, batch_size, train_data, valid_data, embedding_size, heads, depth, seq_length, num_tokens, test_every = 500, test_subset = 10000, test_batchsize = 64, wide = False):

    model = GeneratorTransformer(emb=embedding_size, heads = heads, depth = depth, seq_length = seq_length, num_tokens = num_tokens, wide = wide)


    LOG2E = math.log2(math.e)

    if torch.cuda.is_available():
        model.cuda()

    opt = torch.optim.Adam(lr = 0.001, params = model.parameters())

    for i in trange(epochs):

        opt.zero_grad()

        starts = torch.randint(size = (batch_size, ), low = 0, high = train_data.size(0) - seq_length - 1)
        seqs_source = [train_data[start:start + seq_length] for start in starts]
        seqs_target = [train_data[start+1:start + seq_length + 1] for start in starts]

        source = torch.cat([s[None, :] for s in seqs_source], dim = 0).to(torch.long)
        target = torch.cat([s[None, :] for s in seqs_target], dim = 0).to(torch.long)


        # print(f'Source size: {source.size()}')
        # print(f'Target Size: {target.size()}')

        if torch.cuda.is_available():
            source, target = source.cuda(), target.cuda()

        source, target = Variable(source), Variable(target)

        output = model(source)

        # print(f'Output size {output.size()}')

        loss = F.nll_loss(output.transpose(2, 1), target, reduction = 'mean')

        loss.backward()

        opt.step()

        if i != 0 and (i % test_every == 0 or i == epochs - 1):

            upto = valid_data.size(0) if i == epochs - 1 else test_subset
            data_sub = valid_data[:upto]

            with torch.no_grad():

                bits, tot = 0.0, 0
                batch = []

                for current in range(data_sub.size(0)):

                    fr = max(0, current - seq_length)
                    to = current + 1

                    context = data_sub[fr:to].to(torch.long)

                    if context.size(0) < seq_length + 1:
                        pad = torch.zeros(size = (seq_length + 1 - context.size(0),), dtype = torch.long)
                        context = torch.cat([pad, context], dim = 0)

                        assert context.size(0) == seq_length + 1

                    if torch.cuda.is_available():
                        context = context.cuda()

                    batch.append(context[None, :])

                    if len(batch) == test_batchsize or current == data_sub.size(0) - 1:

                        b = len(batch)

                        all = torch.cat(batch, dim = 0)
                        source = all[:, :-1]
                        target = all[:, -1]

                        output = model(source)

                        lnprobs = output[torch.arange(b, device=d()), -1, target]
                        log2probs = lnprobs * LOG2E

                        bits += -log2probs.sum()
                        batch = []

                bits_per_byte = bits / data_sub.size(0)

                print(f'epoch{i}: {bits_per_byte:.4} bits per byte')

    return model


                




