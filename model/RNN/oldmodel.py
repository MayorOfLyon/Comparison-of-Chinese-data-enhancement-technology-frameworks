import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F

class RNNModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_class):
        super(RNNModel, self).__init__()
        print(embedding_dim, hidden_dim, num_class)
        self.embedding = nn.Embedding(21128, embedding_dim)
        # 双向GRU层
        self.gru1 = nn.GRU(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True)
        # dropout
        self.dropout1 = nn.Dropout(p=0.5)
        # 输出层
        self.output = nn.Linear(hidden_dim*2, num_class)
        nn.init.xavier_normal_(self.embedding.weight)
        nn.init.xavier_normal_(self.output.weight)

    def forward(self, inputs, lengths):
        # 双向GRU
        embeds = self.embedding(inputs)
        x_pack = pack_padded_sequence(embeds, lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        hidden, hn = self.gru1(x_pack)
        hidden, lengths = pad_packed_sequence(hidden, batch_first=True)  
        hidden = self.dropout1(hidden)
        print("Hidden size 1 ",hidden.shape)
        # 全连接
        hidden: torch.Tensor = hidden[:, -1, :].squeeze(1)
        print("Hidden size 2 ",hidden.shape)
        # 输出
        hidden = self.output(hidden)
        print("Hidden size 3 ",hidden.shape)
        log_probs = F.log_softmax(hidden, dim=-1)
        return log_probs