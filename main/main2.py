# coding:utf8
from collections import namedtuple

import torchvision
from torchvision.transforms import transforms

from utils.callback import Runner, AvgStatsCallback

__author__ = 'Sheng Lin'
__date__ = '2020/5/14'
import torch


class MnistLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(256, 32)
        self.linear2 = torch.nn.Linear(32, 10)

    def forward(self, xb):
        xb = xb.view(-1, 16 * 16)
        xb = self.linear1(xb).relu_()
        xb = self.linear2(xb).relu_()
        return xb


class Learner():
    def __init__(self, model, opt, loss_func, data):
        self.model, self.opt, self.loss_func, self.data = model, opt, loss_func, data


def accuracy(out, yb): return (torch.argmax(out, dim=1) == yb).float().mean()


if __name__ == '__main__':
    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
    ])

    train_data = torchvision.datasets.FashionMNIST('~/Code/Machine_Learning/D2L/path/to/imagenet_root/',
                                                   transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=8,
                                               shuffle=True,
                                               num_workers=2)

    vali_data = torchvision.datasets.FashionMNIST('~/Code/Machine_Learning/D2L/path/to/imagenet_root/', train=False,
                                                  transform=transform,
                                                  download=True)
    vali_loader = torch.utils.data.DataLoader(train_data,
                                              batch_size=8,
                                              shuffle=True,
                                              num_workers=2)

    model = MnistLinear()
    learn = Learner(
        model=model,
        opt=torch.optim.Adam(model.parameters()),
        loss_func=torch.nn.CrossEntropyLoss(),
        data=namedtuple('data', ['train_dl', 'valid_dl'])(train_loader, vali_loader),
    )
    acuuracy_cb = AvgStatsCallback(accuracy)
    runner = Runner(acuuracy_cb)
    runner.fit(10, learn)
