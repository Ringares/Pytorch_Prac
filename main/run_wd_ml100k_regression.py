# coding:utf8
from collections import namedtuple
from functools import partial

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

from model.wd import WideAndDeepModel
from utils import callback
# from utils.learner import Learner

__author__ = 'Sheng Lin'
__date__ = '2020/7/2'

data = pd.read_csv('~/Code/Machine_Learning/D2L/data/ml-100k-joined.csv')
used_feature = ['user_id', 'item_id', 'rating',
                'age', 'gender', 'occupation', 'zipcode',
                'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s',
                'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                'Sci-Fi', 'Thriller', 'War', 'Western']

data = data[used_feature]
target = data.pop('rating')

sparse_feat = ['user_id', 'item_id', 'age', 'gender', 'occupation', 'zipcode']
label_encoders = {}
for fname in sparse_feat:
    le = LabelEncoder()
    le.fit(data[fname])
    data[fname] = le.transform(data[fname])
    label_encoders[fname] = (le.classes_.tolist(), dict(zip(le.classes_, le.transform(le.classes_))))

feat_dims = []
for col in data.columns:
    if col in label_encoders:
        feat_dims.append(len(label_encoders[col][0]))
    else:
        feat_dims.append(2)

print(sum(feat_dims))
print(data.max())

train_idx, vali_idx = train_test_split(data.index, test_size=0.2)
train_data = data.iloc[train_idx.values]
vali_data = data.iloc[vali_idx.values]

train_target = target.iloc[train_idx.values]
vali_target = target.iloc[vali_idx.values]


class CSVDataSet(Dataset):
    def __init__(self, data, target):
        self.data = data.values
        self.target = target.values
        # self.target = [1 if i >=3 else 0 for i in target.values]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return torch.tensor(self.data[idx], dtype=torch.long), torch.tensor(self.target[idx], dtype=torch.float32)


train_loader = DataLoader(
    CSVDataSet(train_data, train_target),
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True)

vali_loader = DataLoader(
    CSVDataSet(vali_data, vali_target),
    batch_size=32,
    num_workers=4,
    pin_memory=True)

# X, y = iter(train_loader).next()
# print(X.shape, y.shape, X.mean(), X.std())

################################################################################
#
#
#
################################################################################
dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

learner = callback.Learner(
    model=WideAndDeepModel(feat_dims, 8, [8, 8], dropout=0.2, mode='regression'),
    data=namedtuple('data', ['train_dl', 'valid_dl'])(train_loader, vali_loader),
    loss_func=torch.nn.MSELoss(),
    opt_func=torch.optim.Adam,
    lr=1e-2,
    cbs=[callback.CudaCallback(dev)],
    cb_funcs=[
        partial(callback.AvgStatsCallback, None),
        callback.RecordCallback,
    ]
)

learner.fit(10)