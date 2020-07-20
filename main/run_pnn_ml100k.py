# coding:utf8
from collections import namedtuple
from functools import partial

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

from model.pnn import ProductNeuralNetwork
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
        self.target = [1 if i >=3 else 0 for i in target.values]

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
    model=ProductNeuralNetwork(feat_dims, 8, [8, 8], dropout=0.2, need_inner=True, need_outer=True),
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


epoch 1: train: 0.41949565 valid: 0.39262830 34.5 sec
epoch 1: vali_auc: 0.7775098480036406
epoch 2: train: 0.38314277 valid: 0.37932637 32.7 sec
epoch 2: vali_auc: 0.7958119016605876
epoch 3: train: 0.36606287 valid: 0.38170317 38.2 sec
epoch 3: vali_auc: 0.8000167012623858
epoch 4: train: 0.34685227 valid: 0.37894946 42.6 sec
epoch 4: vali_auc: 0.7964720321123531
epoch 5: train: 0.32611531 valid: 0.39482222 38.6 sec
epoch 5: vali_auc: 0.7867327129137731
epoch 6: train: 0.30747402 valid: 0.40585884 38.0 sec
epoch 6: vali_auc: 0.7861472262796595
epoch 7: train: 0.29363411 valid: 0.41712383 36.3 sec
epoch 7: vali_auc: 0.7814826145357633
epoch 8: train: 0.28257512 valid: 0.43518418 39.0 sec
epoch 8: vali_auc: 0.77832604481804
epoch 9: train: 0.27364331 valid: 0.44747178 48.4 sec
epoch 9: vali_auc: 0.7700176163905753
epoch 10: train: 0.26391821 valid: 0.46966738 52.2 sec
epoch 10: vali_auc: 0.7703010951587367
"""