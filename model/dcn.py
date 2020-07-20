# coding:utf8
import torch

from model.layer import FeaturesEmbedding, MultiLayerPerceptron, CrossNetwork

__author__ = 'Sheng Lin'
__date__ = '2020/7/8'


class DeepAndCrossNetwork(torch.nn.Module):
    """
    Implementation of Deep&CrossNetwork

    Reference:
        Wang, Ruoxi et al. “Deep & Cross Network for Ad Click Predictions.” ArXiv abs/1708.05123 (2017): n. pag.
    """
    def __init__(self, field_dims, embed_dim, num_cn_layer, hidden_dims, dropout, mode='binary', out_dim=1):
        super().__init__()
        self.mode = mode
        self.out_dim = out_dim
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.concat_embed_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.concat_embed_dim, hidden_dims, dropout, output_layer=False)
        self.cross_network = CrossNetwork(self.concat_embed_dim, num_cn_layer)
        self.final_linear = torch.nn.Linear(self.concat_embed_dim+hidden_dims[-1], out_dim)

    def forward(self, x):
        embed = self.embedding(x)
        y_mlp = self.mlp(embed.view(-1, self.concat_embed_dim))
        y_cn = self.cross_network(embed.view(-1, self.concat_embed_dim))
        y = torch.cat((y_cn, y_mlp), dim=1)
        y = self.final_linear(y)
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
    model = DeepAndCrossNetwork(field_dims, 16, 2, [8, 8], 0.2)
    print(model(x))
