import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.autograd import Variable
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score

import math, random 
from tqdm import tqdm, trange


from .ClassificationTransformer import ClassificationTransformer
from .utilities import mask_
from .utilities import device_ as d 



class FitClassification(object):

    def __init__(self, optim_name: str,  embedding_size: int, heads: int, depth: int, seq_length: int, num_tokens: int, num_classes: int, device: str, ff_hidden_mult: int, max_pool = True, dropout = 0.0, mask = True, wide = False):

        self.optim_name = optim_name 
        self.embedding_size = embedding_size
        self.heads = heads 
        self.depth = depth 
        self.seq_length = seq_length
        self.num_tokens = num_tokens 
        self.num_classes = num_classes 
        self.device = device
        self.ff_hidden_mult = ff_hidden_mult
        self.max_pool = max_pool
        self.dropout = dropout
        self.mask = mask 
        self.wide = wide 
        self.model = self.prepare_model()

    def prepare_model(self, pretrained = None):

        if pretrained:
            model = ClassificationTransformer(emb = self.embedding_size, heads = self.heads, depth = self.depth, seq_length=self.seq_length, num_tokens=self.num_tokens, num_classes = self.num_classes, device=self.device, ff_hidden_mult=self.ff_hidden_mult, max_pool=self.max_pool, dropout=self.dropout, wide = self.wide, mask = self.mask)
            model = torch.load(pretrained, map_location=self.device)
            return model 
        else:
            model = ClassificationTransformer(emb = self.embedding_size, heads = self.heads, depth = self.depth, seq_length=self.seq_length, num_tokens=self.num_tokens, num_classes = self.num_classes, device=self.device, ff_hidden_mult=self.ff_hidden_mult, max_pool=self.max_pool, dropout=self.dropout, wide = self.wide, mask = self.mask)
            return model 
    
    def get_optimizer(self, lr: int, momentum: int = None, beta_1: int = None, beta_2: int = None, wd: int = None):

        assert self.optim_name in ['Adam', 'SGD'], f"Only two optimizers are available ['Adam', 'SGD']"

        if self.optim_name == 'Adam':
            if wd:
                return optim.Adam(self.model.parameters(), lr = lr, weight_decay=wd)
            if beta_1 and not beta_2:
                return 'Enter the upper and lower momentum bounds for Adam optimizer `beta_1`, `beta_2`'
            if beta_1 and beta_2:
                return optim.Adam(self.model.parameters(), lr = lr, betas = (beta_1, beta_2))
            if beta_1 and beta_2 and wd:
                return optim.Adam(self.model.parameters(), lr = lr, weight_decay = wd, betas = (beta_1, beta_2))
            if not wd and not beta_1 and not beta_2:
                return optim.Adam(self.model.parameters(), lr = lr)

        if self.optim_name == 'SGD':
            if beta_1 or beta_2:
                return f'SGD takes momentum not the bounds'
            if not wd and not momentum:
                return optim.SGD(self.model.parameters(), lr = lr)
            if wd:
                return optim.SGD(self.model.parameters(), lr = lr, weight_decay=wd)
            if momentum:
                return optim.SGD(self.model.parameters(), lr = lr, momentum=momentum)
            if wd and momentum:
                return optim.SGD(self.model.parameters(), lr = lr, weight_decay=wd, momentum=momentum)


    def fit(self, epochs, dataloader, lr: int, momentum: int = None, beta_1: int = None, beta_2: int = None, wd: int = None, phase = 'training'):

        optimizer = self.get_optimizer(lr = lr, momentum = momentum, beta_1 = beta_1, beta_2 = beta_2, wd = wd)
        
        loss_lst = []
        acc_lst = []
        f1_lst = []
        for epoch in  trange(epochs):
            
            running_loss = 0.0
            true_ = []
            pred_ = []
            
            if phase == 'training':
                self.model.train()

            if phase == 'validation':
                self.model.eval()

            for ix, batch in enumerate(dataloader):

                if phase == 'training':
                    optimizer.zero_grad()

                input_, target = batch[0], batch[1]

                input_, target = input_.to(self.device), target.to(self.device)

                input_, target = Variable(input_), Variable(target)

                output = self.model(input_)

                loss = F.nll_loss(output, target)

                running_loss += loss.item()

                if phase == 'training':

                    loss.backward()
                    optimizer.step()

                preds = torch.argmax(output, dim = 1)

                true_.extend(target.flatten().tolist())
                pred_.extend(preds.flatten().tolist())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_f1_score = f1_score(true_, pred_, average='weighted')
            epoch_acc = accuracy_score(true_, pred_)

            t = f"""
            Epoch: {epoch}, {phase.upper()} phase
                -> Loss: {epoch_loss} \n
                -> Accuracy: {epoch_acc} \n 
                -> F1 Score: {epoch_f1_score}
            """

            print(t)
            loss_lst.append(epoch_loss)
            acc_lst.append(epoch_acc)
            f1_lst.append(epoch_f1_score)

        return loss_lst, acc_lst, f1_lst