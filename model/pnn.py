# coding:utf8
import torch

from model.layer import FeaturesEmbedding, MultiLayerPerceptron, InnerProductLayer, OuterProductLayer

__author__ = 'Sheng Lin'
__date__ = '2020/7/8'


class ProductNeuralNetwork(torch.nn.Module):
    """
    Implementation of PNN

    Reference:
        Qu, Y., Fang, B., Zhang, W., Tang, R., Niu, M., Guo, H., Yu, Y., & He, X. (2019).
        Product-Based Neural Networks for User Response Prediction over Multi-Field Categorical Data.
        ACM Transactions on Information Systems (TOIS), 37, 1 - 35.
    """

    def __init__(self, field_dims, embed_dim, hidden_dims, dropout, need_inner=True, need_outer=False):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.concat_embed_dim = len(field_dims) * embed_dim
        product_size = 0
        if need_inner:
            self.inner_product_layer = InnerProductLayer()
            product_size += len(field_dims) * (len(field_dims) - 1) // 2
        if need_outer:
            self.outer_product_layer = OuterProductLayer(len(field_dims), embed_dim)
            product_size += len(field_dims) * (len(field_dims) - 1) // 2
        self.mlp = MultiLayerPerceptron(self.concat_embed_dim + product_size, hidden_dims, dropout)

    def forward(self, x):
        """
        :param x: [batch_size, field_dims]
        """
        embed = self.embedding(x)  # [b, n, d]
        p_layer = embed.view(-1, self.concat_embed_dim)
        if self.inner_product_layer:
            inner = self.inner_product_layer(embed)  # [b, n(n-1)//2]
            p_layer = torch.cat([p_layer, inner], dim=1)
        if self.outer_product_layer:
            outer = self.outer_product_layer(embed)
            p_layer = torch.cat([p_layer, outer], dim=1)

        y_mlp = self.mlp(p_layer)
        return torch.sigmoid(y_mlp.squeeze())


if __name__ == '__main__':
    field_dims = [3, 4, 5, 6, 3]
    x = torch.randint(1, 3, (4, 5))
    model = ProductNeuralNetwork(field_dims, 16, [8, 8], 0.2, need_outer=True)
    print(model(x))
