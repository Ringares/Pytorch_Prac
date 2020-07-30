# coding:utf8
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from collections import namedtuple
import pandas as pd
import torch
import wandb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

from model.pnn import ProductNeuralNetwork
from utils import callback
# from utils.learner import Learner
from utils.regularization import Regularization

__author__ = 'Sheng Lin'
__date__ = '2020/7/2'

hyperparameter_defaults = dict(
    learning_rate=1e-2,
    batch_size=32,
    epochs=5,
    dropout=0.5,
    embed_dim=8,
    embed_l2=1e-3,
)

wandb.init(project='deep_ctr', config=hyperparameter_defaults)
config = wandb.config

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
        self.target = [1 if i >= 3 else 0 for i in target.values]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return torch.tensor(self.data[idx], dtype=torch.long), torch.tensor(self.target[idx], dtype=torch.float32)


train_loader = DataLoader(
    CSVDataSet(train_data, train_target),
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True)

vali_loader = DataLoader(
    CSVDataSet(vali_data, vali_target),
    batch_size=config.batch_size,
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
    model=ProductNeuralNetwork(feat_dims, config.embed_dim, [16, 16], dropout=config.dropout, need_inner=True,
                               need_outer=True),
    data=namedtuple('data', ['train_dl', 'valid_dl'])(train_loader, vali_loader),
    loss_func=torch.nn.BCELoss(),
    opt_func=torch.optim.Adam,
    lr=config.learning_rate,
    cbs=[callback.CudaCallback(dev),
         callback.RecordCallback(),
         # callback.AvgStatsCallback(None),
         # callback.AucCallback(),
         callback.WandbCallback(metrics=None, need_auc=True, initialized=True)
         ],
    regular=Regularization(weight_decay=config.embed_l2, param_name='embedding.weight')
)

learner.fit(config.epochs)
