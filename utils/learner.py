# coding:utf8
__author__ = 'Sheng Lin'
__date__ = '2020/7/2'


class Learner():
    def __init__(self, model, opt, loss_func, data):
        self.model, self.opt, self.loss_func, self.data = model, opt, loss_func, data
