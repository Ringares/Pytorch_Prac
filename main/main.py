# coding:utf8
import torchvision
import torch
from torchvision.transforms import transforms

__author__ = 'Sheng Lin'
__date__ = '2020/5/14'
import numpy as np
import torch


def loss_batch(model, loss_func, xb, yb, opt=None, device=None):
    if device:
        xb.to(device)
        yb.to(device)
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def acc_batch(model, xb, yb, device=None):
    if device:
        xb.to(device)
        yb.to(device)
    return (torch.argmax(model(xb), dim=1) == yb).float().mean().item(), len(xb)


def fit(epochs, model, loss_func, opt, train_dl, valid_dl, device=None):
    if device:
        model.to(device)
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt, device)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb, device) for xb, yb in valid_dl]
            )
            acc, nums = zip(
                *[acc_batch(model, xb, yb, device) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        val_acc = np.sum(np.multiply(acc, nums)) / np.sum(nums)

        print(epoch, val_loss, val_acc)


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


if __name__ == '__main__':
    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
    ])

    train_data = torchvision.datasets.FashionMNIST('~/Code/Machine_Learning/D2L/path/to/imagenet_root/', transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=8,
                                               shuffle=True,
                                               num_workers=2)

    vali_data = torchvision.datasets.FashionMNIST('~/Code/Machine_Learning/D2L/path/to/imagenet_root/', train=False, transform=transform,
                                                  download=True)
    vali_loader = torch.utils.data.DataLoader(train_data,
                                              batch_size=8,
                                              shuffle=True,
                                              num_workers=2)

    model = MnistLinear()
    opt = torch.optim.Adam(model.parameters())
    loss = torch.nn.CrossEntropyLoss()
    fit(10, model, loss, opt, train_loader, vali_loader)
