# coding:utf8
import torch

__author__ = 'Sheng Lin'
__date__ = '2020/7/20'


class Regularization(torch.nn.Module):
    def __init__(self, weight_decay=1e-5, p=2, param_name='weight'):
        """
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        """
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.weight_decay = weight_decay
        self.p = p
        self.param_name = param_name

    def forward(self, model):
        reg_loss = self.regularization_loss(model, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        """
        获得模型的权重列表
        :param param_name:
        :param model:
        :return:
        """
        weight_list = []
        for name, param in model.named_parameters():
            if self.param_name in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, model, weight_decay, p=2):
        """
        计算张量范数
        :param model:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        """
        reg_loss = 0
        for name, w in self.get_weight(model):
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

    def weight_info(self, model):
        """
        打印权重列表信息
        :param model:
        :return:
        """
        print("---------------regularization weight---------------")
        for name, w in self.get_weight(model):
            print(f'{name}: decay={self.weight_decay} p={self.p}')
        print("---------------------------------------------------")
