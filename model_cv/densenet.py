# coding:utf8

from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F


__author__ = 'Sheng Lin'
__date__ = '2020/6/2'


class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        """
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        """
        super().__init__()
        # bottle neck
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False))
        self.drop_rate = float(drop_rate)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            prev_features = [x]
        else:
            prev_features = x
        x = torch.cat(prev_features, 1)
        x = self.conv1(self.relu1(self.norm1(x)))
        x = self.conv2(self.relu2(self.norm2(x)))
        if self.drop_rate > 0:
            x = F.dropout(x, self.drop_rate)
        return x


class DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, growth_rate, bn_size=4, drop_rate=0):
        super().__init__()
        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, x):
        features = [x]
        for l in self.children():
            y = l(features)
            features.append(y)
        return torch.cat(features, 1)


class Transition(nn.Sequential):
    def __init__(self, num_in_feat, num_out_feat):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_in_feat))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_in_feat, num_out_feat, kernel_size=1, stride=1))
        self.add_module('avgpool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(self, num_init_features, block_config, growth_rate, num_classes=10, bn_size=4, drop_rate=0):
        super().__init__()
        self.features = nn.Sequential(
            OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                ('norm0', nn.BatchNorm2d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('max_pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
                # kernel=3, padding=1, 不能用 AdaptiveMaxPool2d
            ])
        )
        num_features = num_init_features
        for i, num_layer in enumerate(block_config):
            # dense block
            dense_block = DenseBlock(num_layer, num_features, growth_rate, bn_size, drop_rate)
            self.features.add_module('dense_block%d' % (i + 1), dense_block)
            num_features = num_features + num_layer * growth_rate
            # transition layer
            if i != len(block_config) - 1:
                trans_layer = Transition(num_features, num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans_layer)
                num_features = num_features // 2

        # 最后
        self.final = nn.Sequential(
            OrderedDict([
                ('norm5', nn.BatchNorm2d(num_features)),
                ('relu5', nn.ReLU(inplace=True)),
                ('avgpool5', nn.AdaptiveAvgPool2d(1)),
                ('flatten', nn.Flatten(1)),
                ('linear', nn.Linear(num_features, num_classes))
            ])
        )

    def forward(self, x):
        x = self.features(x)
        x = self.final(x)
        return x


def remove_sequential(network, all_layers=None):
    if all_layers is None:
        all_layers = []
    for layer in network.children():
        if isinstance(layer, nn.Sequential): # if sequential layer, apply recursively to layers in sequential layer
            remove_sequential(layer, all_layers)
        if type(layer) == DenseBlock: # if sequential layer, apply recursively to layers in sequential layer
            all_layers.append(layer)
        if list(layer.children()) == []: # if leaf node, add it to list
            all_layers.append(layer)
    return all_layers


def layer_description(model, x):
    for layer in remove_sequential(model):
        x = layer(x)
        print(layer.__class__.__name__, 'Output shape:\t', x.shape)


if __name__ == '__main__':
    model = DenseNet(64, (4, 4, 4), 16)
    print(model)
    layer_description(model, x=torch.randn((1, 3, 224, 224)))
