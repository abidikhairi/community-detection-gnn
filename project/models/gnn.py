import math
import torch as th
import torch.nn as nn


class GraphConvolutionLayer(nn.Module):
    
    def __init__(self, in_feats, out_feats) -> None:
        super().__init__()

        self.out_feats = out_feats
        self.in_feats = in_feats
        
        self.weight = nn.Parameter(th.FloatTensor(in_feats, out_feats))
        self.bias = nn.Parameter(th.FloatTensor(out_feats))

    
    def reset_parameters(self):
        stdv = math.sqrt(self.out_feats)

        self.weight.data.uniform_(-1.0/stdv, 1.0/stdv)
        self.bias.data.uniform_(-1.0/stdv, 1.0/stdv)


    def forward(self, adj, x):
        support = th.mm(adj, x)

        rst = th.matmul(support, self.weight)
        rst = rst + self.bias

        return rst


class GCN(nn.Module):
    def __init__(self, nfeats, nhids, nclasses) -> None:
        super().__init__()

        self.conv1 = GraphConvolutionLayer(nfeats, nhids)
        self.conv2 = GraphConvolutionLayer(nhids, nclasses)


    def forward(self, adj, x):
        x = th.relu(self.conv1(adj, x))
        x = self.conv2(adj, x)

        return x
