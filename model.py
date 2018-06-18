import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CNN_Text(nn.Module):

    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args

        WD = args['word_embed_dim']
        PD = args['pos_embed_dim']
        # D = args['hidden_dim']
        Ci = 1
        Co = args['kernel_num']
        Ks = args['kernel_sizes']

        print('word_num: ', args['words_num'])

        self.word_embed = nn.Embedding(args['words_num'], WD)
        self.pos_embed = nn.Embedding(args['pos_num'], PD)

        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, WD+2*PD)) for K in Ks])

        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(args['dropout'])
        self.fc1 = nn.Linear(len(Ks) * Co, 6)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, tokens, pos1, pos2):

        tokens = self.word_embed(tokens)  # (N, W, D)
        pos1 = self.pos_embed(pos1)
        pos2 = self.pos_embed(pos2)

        if self.args['static']:
            tokens = Variable(tokens)

        tokens = torch.cat((tokens, pos1, pos2), 2)

        tokens = tokens.unsqueeze(1)  # (N, Ci, W, D)

        tokens = [F.relu(conv(tokens)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        tokens = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in tokens]  # [(N, Co), ...]*len(Ks)

        tokens = torch.cat(tokens, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        tokens = self.dropout(tokens)  # (N, len(Ks)*Co)
        logit = self.fc1(tokens)  # (N, C)
        return logit
