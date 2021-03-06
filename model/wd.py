# coding:utf8
import torch

from model.layer import FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron

__author__ = 'Sheng Lin'
__date__ = '2019/11/29'


class WideAndDeepModel(torch.nn.Module):
    """
    Implementation of Wide&Deep

    Reference:
        HT Cheng, et al. Wide & Deep Learning for Recommender Systems, 2016.
    """
    def __init__(self, field_dims, embed_dim, hidden_dims, dropout, mode='binary', out_dim=1):
        super().__init__()
        self.mode = mode
        self.out_dim = out_dim
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.concat_embed_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.concat_embed_dim, hidden_dims, dropout)

    def forward(self, x):
        y_linear = self.linear(x)
        embed = self.embedding(x)
        y_mlp = self.mlp(embed.view(-1, self.concat_embed_dim))
        y = y_linear + y_mlp

        if self.mode == 'binary' and self.out_dim == 1:
            return torch.sigmoid(y.squeeze(1))
        elif self.mode == 'multiclass' and self.out_dim > 1:
            return torch.softmax(y, self.out_dim)
        elif self.mode == 'regression':
            return y.squeeze(1)
        else:
            raise ValueError('unimplemented mode: ' + self.mode)


if __name__ == '__main__':
    field_dims = [3, 4, 5, 6, 3]
    x = torch.randint(1, 3, (4, 5))
    model = WideAndDeepModel(field_dims, 16, [8, 8], 0.2)
    print(model(x))
