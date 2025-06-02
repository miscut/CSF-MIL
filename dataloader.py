'''
dataloader
'''

import numpy as np
import os
import torch
from torch.utils.data import Dataset

def label_separate(training_list):
    list_0 = []
    list_1 = []
    for tl in training_list:
        tl_label = tl.split(',')[1].replace("]", '')
        if int(tl_label) == 0:
            list_0.append(tl)
        else:
            list_1.append(tl)
    return list_0,list_1

def read_bag(train_list, data_path):
    patient_list = []
    label_list = []
    for t in train_list:  # patient level,eg:['3', 0],<class 'str'>
        patient = (t.split(',')[0].replace("[\'", '')).replace("\'", '')
        patient_path = os.path.join(data_path,'pt_files', patient+'.pt')
        label = t.split(',')[1].replace("]", '')
        patient_list.append(patient_path)
        label_list.append(label)
    return patient_list,label_list

class m3MILDataset(Dataset):
    def __init__(self, df, transform=None):
        super(m3MILDataset, self).__init__()
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        bag_path_a = row['bag_a']
        bag_path_b = row['bag_b']
        bag_path_c = row['bag_c']

        features_a = torch.load(bag_path_a)
        features_b = torch.load(bag_path_b)
        features_c = torch.load(bag_path_c)

        label = row['label']
        label = np.array(int(label))
        label = torch.from_numpy(label)
        label = label.long()

        return features_a, features_b, features_c, label