# coding:utf8
import math
import re
import time
import typing
from collections import OrderedDict
from functools import partial

import torch
import wandb
from sklearn.metrics import roc_auc_score

from utils.regularization import Regularization

__author__ = 'Sheng Lin'
__date__ = '2020/5/14'

_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re2 = re.compile('([a-z0-9])([A-Z])')


class CancelTrainException(Exception): pass


class CancelEpochException(Exception): pass


class CancelBatchException(Exception): pass


def camel2snake(name):
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()


def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, typing.Iterable): return list(o)
    return [o]


def param_getter(m): return m.parameters()


class Learner():
    def __init__(self, model, data, loss_func, opt_func=torch.optim.Adam, lr=1e-2, splitter=param_getter,
                 cbs=None, cb_funcs=None, regular=None):
        self.model, self.data, self.loss_func = model, data, loss_func
        self.opt_func, self.lr, self.splitter = opt_func, lr, splitter
        self.in_train, self.logger, self.opt = False, print, None
        self.regular = regular
        if self.regular:
            self.regular.weight_info(self.model)

        # NB: Things marked "NEW" are covered in lesson 12
        # NEW: avoid need for set_runner
        self.cbs = []
        self.add_cb(TrainEvalCallback())
        self.add_cbs(cbs)
        self.add_cbs(cbf() for cbf in listify(cb_funcs))

    def add_cbs(self, cbs):
        for cb in listify(cbs): self.add_cb(cb)

    def add_cb(self, cb):
        cb.set_runner(self)
        setattr(self, cb.name, cb)
        self.cbs.append(cb)

    def remove_cbs(self, cbs):
        for cb in listify(cbs): self.cbs.remove(cb)

    def one_batch(self, i, xb, yb):
        try:
            self.iter = i
            self.xb, self.yb = xb, yb
            self('begin_batch')
            self.pred = self.model(self.xb)
            self('after_pred')
            self.loss = self.loss_func(self.pred, self.yb)
            # add additional regularization (to device)
            if self.regular and isinstance(self.regular, Regularization):
                self.loss = self.loss + self.regular(self.model)
            self('after_loss')
            if not self.in_train: return
            self.loss.backward()
            self('after_backward')
            self.opt.step()
            self('after_step')
            self.opt.zero_grad()
        except CancelBatchException:
            self('after_cancel_batch')
        finally:
            self('after_batch')

    def all_batches(self):
        self.iters = len(self.dl)
        try:
            for i, (xb, yb) in enumerate(self.dl): self.one_batch(i, xb, yb)
        except CancelEpochException:
            self('after_cancel_epoch')

    def do_begin_fit(self, epochs):
        self.epochs, self.loss = epochs, torch.tensor(0.)
        self('begin_fit')

    def do_begin_epoch(self, epoch):
        self.epoch, self.dl = epoch, self.data.train_dl
        return self('begin_epoch')

    def fit(self, epochs, cbs=None, reset_opt=False):
        # NEW: pass callbacks to fit() and have them removed when done
        self.add_cbs(cbs)
        # NEW: create optimizer on fit(), optionally replacing existing
        if reset_opt or not self.opt: self.opt = self.opt_func(self.splitter(self.model), lr=self.lr)

        try:
            self.do_begin_fit(epochs)
            for epoch in range(epochs):
                if not self.do_begin_epoch(epoch):
                    self.all_batches()

                with torch.no_grad():
                    self.dl = self.data.valid_dl
                    if not self('begin_validate'):
                        self.all_batches()
                self('after_epoch')

        except CancelTrainException:
            self('after_cancel_train')
        finally:
            self('after_fit')
            self.remove_cbs(cbs)

    ALL_CBS = {'begin_batch', 'after_pred', 'after_loss', 'after_backward', 'after_step',
               'after_cancel_batch', 'after_batch', 'after_cancel_epoch', 'begin_fit',
               'begin_epoch', 'begin_validate', 'after_epoch',
               'after_cancel_train', 'after_fit'}

    def __call__(self, cb_name):
        res = False
        assert cb_name in self.ALL_CBS
        for cb in sorted(self.cbs, key=lambda x: x._order): res = cb(cb_name) and res
        return res


class Callback:
    _order = 0

    def set_runner(self, run): self.run = run

    def __getattr__(self, k): return getattr(self.run, k)

    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return camel2snake(name or 'callback')

    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        if f and f(): return True
        return False


class TrainEvalCallback(Callback):
    def begin_fit(self):
        self.run.n_epochs = 0.
        self.run.n_iter = 0

    def begin_epoch(self):
        self.run.n_epochs = self.epoch
        self.model.train()
        self.run.in_train = True

    def after_batch(self):
        if not self.in_train:
            return
        self.run.n_epochs += 1. / self.iters
        self.run.n_iter += 1

    def begin_validate(self):
        self.model.eval()
        self.run.in_train = False


class AvgStats():
    def __init__(self, metrics, in_train):
        self.metrics, self.in_train = listify(metrics), in_train

    def reset(self):
        self.tot_loss, self.count = 0., 0
        self.tot_mets = [0.] * len(self.metrics)

    @property
    def all_stats(self):
        return [self.tot_loss.item()] + [i.item() for i in self.tot_mets]

    @property
    def avg_stats(self):
        return [o / self.count for o in self.all_stats]

    def __repr__(self):
        if not self.count: return ""
        return f"{'train' if self.in_train else 'valid'}: {' '.join([f'{i:.8f}' for i in self.avg_stats])}"

    def accumulate(self, run):
        bn = run.xb.shape[0]
        self.tot_loss += run.loss * bn
        self.count += bn
        for i, m in enumerate(self.metrics):
            self.tot_mets[i] += m(run.pred, run.yb) * bn


class AucCallback(Callback):
    def __init__(self):
        self.epoch_pred = []
        self.epoch_y = []

    def begin_validate(self):
        self.epoch_pred = []
        self.epoch_y = []

    def after_pred(self):
        self.epoch_pred += self.run.pred.cpu().data.flatten().numpy().tolist()
        self.epoch_y += self.run.yb.cpu().numpy().tolist()

    def after_epoch(self):
        auc = roc_auc_score(self.epoch_y, self.epoch_pred)
        print(f"epoch {self.epoch+1}: vali_auc: {auc}")


class AvgStatsCallback(Callback):
    def __init__(self, metrics, need_time=True):
        self.train_stats = AvgStats(metrics, True)
        self.valid_stats = AvgStats(metrics, False)
        self.need_time = need_time

    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()
        self.run.epoch_ts = time.time()

    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad():
            stats.accumulate(self.run)

    def after_epoch(self):
        self.run.epoch_ts = time.time() - self.run.epoch_ts
        time_str = f"{self.run.epoch_ts:.1f} sec" if self.need_time and self.run.epoch_ts else ''
        print(f"epoch {self.epoch+1}: {self.train_stats} {self.valid_stats} {time_str}")


class RecordCallback(Callback):
    def begin_fit(self):
        self.lrs = []
        self.losses = []

    def after_batch(self):
        self.lrs.append(self.opt.param_groups[-1]['lr'])
        self.losses.append(self.loss.detach().cpu())


class CudaCallback(Callback):
    def __init__(self, device):
        self.device = device

    def begin_fit(self):
        self.model.to(self.device)

    def begin_batch(self):
        self.run.xb = self.xb.to(self.device)
        self.run.yb = self.yb.to(self.device)


class WandbCallback(Callback):
    """
    if metrics is None and need_auc==Flase:
    then only loss will be considered
    """

    def __init__(self, metrics, need_auc=False, need_time=True, proj_name='Default', verbose=True, config=None, initialized=False):
        if not initialized:
            wandb.init(project=proj_name, config=config)

        self.train_stats = AvgStats(metrics, True)
        self.valid_stats = AvgStats(metrics, False)
        self.train_stats_name = ['train_loss'] + (['train_' + i.__name__ for i in metrics] if metrics else [])
        self.vali_stats_name = ['vali_loss'] + (['vali_' + i.__name__ for i in metrics] if metrics else [])

        self.need_time = need_time
        self.verbose = verbose
        self.need_auc = need_auc
        if self.need_auc:
            self.epoch_pred = []
            self.epoch_y = []

    def begin_fit(self):
        wandb.watch(self.model)

    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()
        self.run.epoch_ts = time.time()

    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad():
            stats.accumulate(self.run)

    def begin_validate(self):
        if self.need_auc:
            self.epoch_pred = []
            self.epoch_y = []

    def after_pred(self):
        if self.need_auc:
            self.epoch_pred += self.run.pred.cpu().data.flatten().numpy().tolist()
            self.epoch_y += self.run.yb.cpu().numpy().tolist()

    def after_epoch(self):
        auc = roc_auc_score(self.epoch_y, self.epoch_pred) if self.need_auc else None
        if self.verbose:
            self.run.epoch_ts = time.time() - self.run.epoch_ts
            time_str = f"{self.run.epoch_ts:.1f} sec" if self.need_time and self.run.epoch_ts else ''
            print(f"epoch {self.epoch}: {self.train_stats} {self.valid_stats} {time_str}")
            if auc:
                print(f"epoch {self.epoch}: vali_auc: {auc}")
        logs = OrderedDict({'epoch': self.epoch})
        logs.update(zip(self.train_stats_name, self.train_stats.avg_stats))
        logs.update(zip(self.vali_stats_name, self.valid_stats.avg_stats))
        if auc:
            logs['auc'] = auc
            # wandb.plots.roc.roc(self.epoch_y, self.epoch_pred)
        # print(logs)
        wandb.log(logs)


class BatchTransformXCallback(Callback):
    _order = 2

    def __init__(self, tfm):
        self.tfm = tfm

    def begin_batch(self):
        self.run.xb = self.tfm(self.run.xb)


class ParamScheduler(Callback):
    _order = 1

    def __init__(self, pname, sched_funcs):
        self.pname, self.sched_funcs = pname, sched_funcs

    def begin_fit(self):
        if not isinstance(self.sched_funcs, (list, tuple)):
            self.sched_funcs = [self.sched_funcs] * len(self.opt.param_groups)

    def set_param(self):
        assert len(self.opt.param_groups) == len(self.sched_funcs)
        for pg, f in zip(self.opt.param_groups, self.sched_funcs):
            pg[self.pname] = f(self.n_epochs / self.epochs)

    def begin_batch(self):
        if self.in_train: self.set_param()


def annealer(f):
    def _inner(start, end): return partial(f, start, end)

    return _inner


@annealer
def sched_lin(start, end, pos): return start + pos * (end - start)


@annealer
def sched_cos(start, end, pos): return start + (1 + math.cos(math.pi * (1 - pos))) * (end - start) / 2


@annealer
def sched_no(start, end, pos): return start


@annealer
def sched_exp(start, end, pos): return start * (end / start) ** pos


def combine_scheds(pcts, scheds):
    assert sum(pcts) == 1.
    pcts = torch.tensor([0] + listify(pcts))
    assert torch.all(pcts >= 0)
    pcts = torch.cumsum(pcts, 0)

    def _inner(pos):
        idx = (pos >= pcts).nonzero().max()
        actual_pos = (pos - pcts[idx]) / (pcts[idx + 1] - pcts[idx])
        return scheds[idx](actual_pos)

    return _inner
