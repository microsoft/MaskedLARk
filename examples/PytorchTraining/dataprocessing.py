from torch.utils.data import Dataset
import torch
from sklearn import datasets
import numpy as np
import random


class BreastCancerDemoDataset(Dataset):
    def __init__(self, split='train'):
        all_data = datasets.load_breast_cancer()
        indices = list(range(569))
        # random.shuffle(indices)

        if split=='train':
            indices = indices[:500]
        elif split=='val':
            indices = indices[500:]
        self.dataset = all_data.data[indices,:]
        self.targets = all_data.target[indices]

    def __len__(self):
        return len(self.dataset[:, 0])

    def __getitem__(self, index):
        # datapoint = self.dataset[index,:]
        datapoint = np.round(self.dataset[index, :]*1000)
        target = self.targets[index]
        return datapoint, target


def main():
    bc_dataset = BreastCancerDemoDataset()
    print(bc_dataset[0])

if __name__=="__main__":
    main()

