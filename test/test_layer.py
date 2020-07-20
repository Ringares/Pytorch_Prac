# coding:utf8
import torch
from model.layer import FeaturesLinear, FeaturesEmbedding, CrossNetwork, InnerProductLayer, OuterProductLayer

__author__ = 'Sheng Lin'
__date__ = '2020/7/2'


def test_FeaturesLinear():
    field_dims = [3, 4, 5, 6]
    linear = FeaturesLinear(field_dims)
    cols = []
    for i in field_dims:
        cols.append(torch.randint(0, i, (16, 1)))
    x = torch.cat(cols, dim=1)

    print(x, x.shape)  # (16, 4) (idx_in_batch, idx_in_field)
    print(linear(x), linear(x).shape)  # (16,1)


def test_FeaturesEmbedding():
    field_dims = [3, 4, 5, 6]
    emb = FeaturesEmbedding(field_dims, 8)
    cols = []
    for i in field_dims:
        cols.append(torch.randint(0, i, (16, 1)))
    x = torch.cat(cols, dim=1)

    print(x, x.shape)  # (16, 4) (idx_in_batch, idx_in_field)
    print(emb(x), emb(x).shape)  # (16,8)


def test_CrossNetwork():
    x = torch.randint(1, 10, (2, 5), dtype=torch.float32)
    print(x)
    layer = CrossNetwork(5, 2)
    print(layer(x), layer(x).shape)


def test_InnerProductLayer():
    x = torch.randint(1, 10, (2, 5, 4), dtype=torch.float32)
    print(x)
    layer = InnerProductLayer()
    print(layer(x), layer(x).shape)


def test_OuterProductLayer():
    x = torch.randint(1, 10, (2, 5, 4), dtype=torch.float32)
    print(x)
    layer = OuterProductLayer(5, 4)
    print('mat', layer(x), layer(x).shape)
    layer = OuterProductLayer(5, 4, 'vec')
    print('vec', layer(x), layer(x).shape)
    layer = OuterProductLayer(5, 4, 'num')
    print('num', layer(x), layer(x).shape)
