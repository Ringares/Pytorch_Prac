# coding:utf8
from collections import OrderedDict

import numpy as np
import torch
from torch import nn


class FeaturesLinear(nn.Module):
    def __init__(self, field_dims, output_dim=1):
        """
        用一维 embedding 模拟线性函数 f(x) = wx+b
        只能接受特征的离散 label encode; 且需要知道每个特征 encoder 的长度
        计算每个特征对应的 offset 起始位置
        :param field_dims:
        :param output_dim:
        """
        super().__init__()
        self.fc = nn.Embedding(sum(field_dims), output_dim)
        self.bias = nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


class FeaturesEmbedding(nn.Module):
    def __init__(self, field_dims, embed_dim):
        """
        用一个大的 embed 矩阵, 对所有特征进行 embedding,
        需要 计算每个特征对应的 offset 起始位置
        只能接受特征的离散 label encode; 且需要知道每个特征 encoder 的长度

        :param field_dims:
        :param embed_dim:
        """
        super().__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class FactorizationMachine(nn.Module):
    def __init__(self, reduce_sum=True):
        """
        计算 FM 的交叉部分,
        :param reduce_sum:
        """
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout, output_layer=True):
        super().__init__()
        layers = OrderedDict()
        for idx, layer_dim in enumerate(hidden_dims):
            layers['mlp_%s_linear' % idx] = nn.Linear(input_dim, layer_dim)
            layers['mlp_%s_bn' % idx] = nn.BatchNorm1d(layer_dim)
            layers['mlp_%s_relu' % idx] = nn.ReLU()
            layers['mlp_%s_dropout' % idx] = nn.Dropout(p=dropout)
            input_dim = layer_dim
        if output_layer:
            layers['mlp_last_linear'] = nn.Linear(input_dim, 1)
        self.mlp = nn.Sequential(layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, concat_embed_dim)``
        """
        return self.mlp(x)


class CrossNetwork(nn.Module):
    """
    Implementation of CrossNetwork proposed in Deep&CrossNetwork

    Reference:
        Wang, Ruoxi et al. “Deep & Cross Network for Ad Click Predictions.” ArXiv abs/1708.05123 (2017): n. pag.
    """

    def __init__(self, dim, num_layers):
        super().__init__()
        self.w = nn.ModuleList([nn.Linear(dim, 1) for _ in range(num_layers)])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros((dim,))) for _ in range(num_layers)])

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, concat_embed_dim)``
        """
        x0 = x
        for i, w in enumerate(self.w):
            xw = w(x)
            x = x0 * xw + self.b[i] + x
        return x


class InnerProductLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        idx1, idx2 = [], []
        for i in range(num_fields):
            for j in range(i + 1, num_fields):
                idx1.append(i)
                idx2.append(j)
        # print(idx1, idx2)
        p = x[:, idx1, :]  # [b,k,d]
        q = x[:, idx2, :]  # [b,k,d]
        return torch.sum(p * q, dim=-1)  # [b,k]


class OuterProductLayer(nn.Module):
    def __init__(self, num_field, embed_dim, mode='mat'):
        super().__init__()
        k = num_field * (num_field - 1) // 2
        self.mode = mode
        if mode == 'mat':
            kernel_size = k, embed_dim, embed_dim
        elif mode == 'vec':
            kernel_size = k, embed_dim
        elif mode == 'num':
            kernel_size = k, 1
        else:
            raise ValueError('not valid mode')

        self.kernel = nn.Parameter(torch.zeros(kernel_size))
        nn.init.xavier_uniform_(self.kernel.data)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        idx1, idx2 = [], []
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                idx1.append(i), idx2.append(j)
        p, q = x[:, idx1, :], x[:, idx2, :]  # p,q=[b,k,d] k=n(n-1)//2

        if self.mode == 'mat':
            kp = p.unsqueeze(2) @ self.kernel @ q.unsqueeze(-1)  # [b,k,1,d]@[k,d,d]@[b,k,d,1] = [b,k,1,1]
            kp = kp.squeeze()  # [b,k]
            return kp
        else:
            kp = torch.sum(p * q * self.kernel, dim=-1)
        return kp
