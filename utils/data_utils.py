import csv
import linecache
import subprocess
import mmh3

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class LazyTextDataset(Dataset):
    def __init__(self, filename, skip=0):
        self._filename = filename
        self._skip = skip
        self._total_data = 0
        self._total_data = int(subprocess.check_output("wc -l " + filename, shell=True).split()[0])

    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1 + self._skip)
        csv_line = csv.reader([line])
        return idx + 1 + self._skip, next(csv_line)

    def __len__(self):
        return self._total_data - self._skip


class AvazuDataSet(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path, index_col=False, sep=',')
        self.len = len(self.data)
        lable = 'click'
        sparse_features = ['id', 'hour', 'C1', 'banner_pos',
                           'site_id', 'site_domain', 'site_category',
                           'app_id', 'app_domain', 'app_category',
                           'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type',
                           'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
        # print(self.data.columns)
        self.feat_dims = []
        for feat in sparse_features:
            lbe = LabelEncoder()
            self.data[feat] = lbe.fit_transform(self.data[feat])
            self.feat_dims.append(len(lbe.classes_))

        self.features = self.data[sparse_features]
        self.targets = self.data[lable]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return torch.tensor(self.features.iloc[index].to_numpy(), dtype=torch.long), torch.tensor(
            self.targets.iloc[index], dtype=torch.float32)

    def get_feat_dims(self):
        return self.feat_dims


class AvazuHashDataSet(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path, index_col=False, sep=',')
        self.len = len(self.data)
        lable = 'click'
        sparse_features = ['id', 'hour', 'C1', 'banner_pos',
                           'site_id', 'site_domain', 'site_category',
                           'app_id', 'app_domain', 'app_category',
                           'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type',
                           'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
        # print(self.data.columns)
        self.feat_dims = []
        for feat in sparse_features:
            lbe = LabelEncoder()
            uni_cnt = len(self.data[feat].unique())
            if uni_cnt < 10000:
                self.data[feat] = lbe.fit_transform(self.data[feat])
                self.feat_dims.append(len(lbe.classes_))
            else:
                self.data[feat] = self.data[feat].apply(lambda x: mmh3.hash(str(x), signed=False) % 10000)
                self.feat_dims.append(10000)

        self.features = self.data[sparse_features]
        self.targets = self.data[lable]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return torch.tensor(self.features.iloc[index].to_numpy(), dtype=torch.long), torch.tensor(
            self.targets.iloc[index], dtype=torch.float32)

    def get_feat_dims(self):
        return self.feat_dims

