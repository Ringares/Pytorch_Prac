# coding:utf8
import torch

__author__ = 'Sheng Lin'
__date__ = '2020/7/3'


def accuracy(out, yb):
    return (torch.argmax(out, dim=1) == yb).float().mean()
