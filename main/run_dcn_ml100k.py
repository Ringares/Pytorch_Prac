# coding:utf8
# from utils.learner import Learner
import os
from collections import namedtuple
from functools import partial
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

from model.dcn import DeepAndCrossNetwork
from utils import callback

__author__ = 'Sheng Lin'
__date__ = '2020/7/2'

data = pd.read_csv(Path(os.path.dirname(__file__)) / '../data/ml-100k-joined.csv')
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
        self.target = [1 if i >= 3 else 0 for i in target.values]

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
    model=DeepAndCrossNetwork(feat_dims, 8, 4, [8, 8], dropout=0.2, mode='binary'),
    data=namedtuple('data', ['train_dl', 'valid_dl'])(train_loader, vali_loader),
    loss_func=torch.nn.BCELoss(),
    opt_func=torch.optim.Adam,
    lr=1e-2,
    cbs=[callback.CudaCallback(dev)],
    cb_funcs=[
        partial(callback.AvgStatsCallback, None),
        callback.RecordCallback,
        callback.AucCallback
    ]
)

learner.fit(10)

"""
epoch 1: train: 0.40409917 valid: 0.38419302 23.3 sec
epoch 1: vali_auc: 0.7817918242246089
epoch 2: train: 0.37575708 valid: 0.38517537 22.4 sec
epoch 2: vali_auc: 0.7851001146536177
epoch 3: train: 0.37080813 valid: 0.38341304 22.3 sec
epoch 3: vali_auc: 0.7829418669752125
epoch 4: train: 0.36751504 valid: 0.38693718 23.0 sec
epoch 4: vali_auc: 0.7895630507538034
epoch 5: train: 0.37511736 valid: 0.37896504 20.1 sec
epoch 5: vali_auc: 0.788904441559058
epoch 6: train: 0.36473289 valid: 0.38322698 21.6 sec
epoch 6: vali_auc: 0.7889786973703453
epoch 7: train: 0.36303127 valid: 0.38598931 18.9 sec
epoch 7: vali_auc: 0.7901107373937926
epoch 8: train: 0.37342185 valid: 0.38506836 20.3 sec
epoch 8: vali_auc: 0.7865382097398982
epoch 9: train: 0.35348533 valid: 0.39375872 21.8 sec
epoch 9: vali_auc: 0.7870593491007406
epoch 10: train: 0.35792937 valid: 0.39486299 22.0 sec
epoch 10: vali_auc: 0.786269552319075
"""
