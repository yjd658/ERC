import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch_geometric.utils import softmax
from torch_scatter import scatter_add

import pickle
import torch.nn.functional as F
import math
from modules import transformer
import numpy as np
import random

device = torch.device('cuda:0')
Splen = 60
Spencoder = 'direct'
class AttitudeCNN(nn.Module):
    def __init__(self):
        super(AttitudeCNN, self).__init__()
        ci = 1  # input chanel size
        kernel_num = 50# output chanel size
        kernel_size = [3, 5]
        # vocab_size = param['vocab_size']
        embed_dim = 100
        dropout = 0.5
        # class_num = param['class_num']
        # self.param = param
        # self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.conv11 = nn.Conv2d(ci, kernel_num, (kernel_size[0], embed_dim))
        self.conv12 = nn.Conv2d(ci, kernel_num, (kernel_size[1], embed_dim))
        # self.conv13 = nn.Conv2d(ci, kernel_num, (kernel_size[2], embed_dim))
        self.dropout = nn.Dropout(dropout)
        # self.fc1 = nn.Linear(len(kernel_size) * kernel_num, class_num)

    # def init_embed(self, embed_matrix):
    #     self.embed.weight = nn.Parameter(torch.Tensor(embed_matrix))

    @staticmethod
    def conv_and_pool(x, conv):
        # x: (batch, 1, sentence_length,  )
        x = conv(x)
        # x: (batch, kernel_num, H_out, 1)
        x = F.relu(x.squeeze(3))
        # x: (batch, kernel_num, H_out)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        #  (batch, kernel_num)
        return x

    def forward(self, x):
        # x: (batch, sentence_length)
        # x = self.embed(x)
        # x: (batch, sentence_length, embed_dim)
        # TODO init embed matrix with pre-trained
        x = x.unsqueeze(1)
        # x: (batch, 1, sentence_length, embed_dim)
        x1 = self.conv_and_pool(x, self.conv11)  # (batch, kernel_num)
        x2 = self.conv_and_pool(x, self.conv12)  # (batch, kernel_num)
        # x3 = self.conv_and_pool(x, self.conv13)  # (batch, kernel_num)
        x = torch.cat((x1, x2), 1)  # (batch, 3 * kernel_num)
        x = self.dropout(x)
        # logit = F.log_softmax(self.fc1(x), dim=1)
        return x



class AVECModel(nn.Module):

    def __init__(self,  attr, listener_state=False,
                  dropout_rec=0.5, dropout=0.5):
        super(AVECModel, self).__init__()


        self.attr = attr
        self.dropout = nn.Dropout(dropout)
        self.dropout_rec = nn.Dropout(dropout)
        self.dialog_avec = model = DialogueINAB(base_model='LSTM',
                        base_layer=2,
                        input_size=100,
                        hidden_size=200,
                        n_speakers=2,
                        n_classes=6,
                        dropout=0.2,
                        cuda_flag=True,
                        reason_steps=2)
        self.linear = nn.Linear(920*1, 920*1)
        self.smax_fc = nn.Linear(920*1, 1)

    def forward(self, U, qmask,seq_lengths):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """

        emotions = self.dialog_avec(U, qmask,seq_lengths)  # seq_len, batch, D_e
        emotions = self.dropout_rec(emotions)
        hidden = torch.tanh(self.linear(emotions))
        hidden = self.dropout(hidden)
        if self.attr != 4:
            pred = (self.smax_fc(hidden).squeeze())  # seq_len, batch
        else:
            pred = (self.smax_fc(hidden).squeeze())  # seq_len, batch
        return pred.transpose(0, 1).contiguous().view(-1)





class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len
        target -> batch*seq_len
        mask -> batch*seq_len
        """
        loss = self.loss(pred * mask, target) / torch.sum(mask)
        return loss


if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    ByteTensor = torch.cuda.ByteTensor

else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    ByteTensor = torch.ByteTensor




class DialogueINAB(nn.Module):
    def __init__(self, base_model='LSTM', base_layer=2, input_size=None, hidden_size=None, n_speakers=2,
                 n_classes=6, dropout=0.2, cuda_flag=False, reason_steps=None):

        super(DialogueINAB, self).__init__()
        self.base_model = base_model
        self.n_speakers = n_speakers

        if self.base_model == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=base_layer,
                               bidirectional=True, dropout=dropout)
            self.rnn_parties = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=base_layer,
                                       bidirectional=True, dropout=dropout)
        elif self.base_model == 'GRU':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=base_layer, bidirectional=True,
                              dropout=dropout)
            self.rnn_parties = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=base_layer,
                                      bidirectional=True, dropout=dropout)
        elif self.base_model == 'Linear':
            self.base_linear = nn.Linear(input_size, 2 * hidden_size)
        else:
            print('Base model must be one of LSTM/GRU/Linear')
            raise NotImplementedError


        self.translayer = transformer.TransformerEncoder(400 + Splen, 5, 1)
        # self.attitide=AttitudeCNN()
        # self.multiattn = nn.MultiheadAttention(400, 1, dropout=dropout)
        # self.multiattn = trans.Transformer()

    def forward(self, U, qmask, seq_lengths):
       
        U_s, U_p = None, None
        # H=D.dif(U,qmask)
        if self.base_model == 'LSTM':
            # (b,l,h), (b,l,p)
            U_, qmask_ = U.transpose(0, 1), qmask.transpose(0, 1)  # U_；32,110,100；qmask_:32,110,2
            U_p_ = torch.zeros(U_.size()[0], U_.size()[1], 400).type(U.type())  # U_p:32,110,200
            U_parties_ = [torch.zeros_like(U_).type(U_.type()) for _ in
                          range(self.n_speakers)]  # U_parties:2,32,110,100
            for b in range(U_.size(0)):  # b=1--32次
                for p in range(len(U_parties_)):  # 2次 p=0 or 1
                    index_i = torch.nonzero(qmask_[b][:, p]).squeeze(
                        -1)  # p=0,index_i
                    if index_i.size(0) > 0:
                        U_parties_[p][b][:index_i.size(0)] = U_[b][
                            index_i]  
            # AT=[(self.attitide(U_parties_[p])).unsqueeze(1) for p in range(len(U_parties_))]

            # for p in range(len(U_parties_)):
            #     AT[p]=torch.repeat_interleave(AT[p].unsqueeze(dim=1), repeats=U_parties_[p].size(1), dim=1)
            #     AT[p]=AT[p].squeeze()
            #     U_parties_[p]=torch.cat((U_parties_[p],AT[p]),dim=2)
            E_parties_ = [self.rnn_parties((U_parties_[p]).transpose(0, 1))[0].transpose(0, 1) for p in range(len(
                U_parties_))]  # U_parties_[p]:32,110,100|U_parties_[p].transpose(0, 1):110,32,100|E_parties_:男/女人说话的序列编码

            for b in range(U_p_.size(0)):
                for p in range(len(U_parties_)):
                    index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                    if index_i.size(0) > 0: U_p_[b][index_i] = E_parties_[p][b][:index_i.size(0)]
            U_p = U_p_.transpose(0, 1)

            # (l,b,2*h) [(2*bi,b,h) * 2]

            U_s, hidden = self.rnn(U)  # U:110,32,100 U_s:110,32,200


            Spcode = torch.ones(qmask.size(0), qmask.size(1), Splen).to(device)
            if Spencoder == 'direct':
                for i in range(qmask.size(0)):
                    for j in range(qmask.size(1)):
                        if qmask[i, j, 0] == 0:
                            Spcode[i, j, :] = 0
            # U_s1=U_s
            # U_p1=U_p
            # U_p = self.multiattn(U_p)
            U_s = torch.cat((U_s, Spcode), dim=2)
            U_p = torch.cat((U_p, Spcode), dim=2)

            a = U.ne(0)
            c = torch.cat((a, a, a, a, a), dim=2)
            c = c[:, :, 0:400 + Splen]
            U_s = U_s * c
            U_p = U_p * c



            U_l = self.translayer(seq_lengths, U_s, U_p, U_p)
            U_n = self.translayer(seq_lengths, U_p, U_s, U_s)

            # U_s = U_s.transpose(0, 1)

            U_ln=torch.cat((U_n,U_p),dim=-1)
        


        return U_ln



