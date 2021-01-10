from torch.utils.data import Dataset
import numpy as np
import torch


class NumpyDataSet(Dataset):

    def __init__(self, X_path, Y_path):
        super(NumpyDataSet, self).__init__()
        self.X = np.load(X_path)
        self.Y = np.load(Y_path)

    def __getitem__(self, index):
        return torch.cuda.FloatTensor([self.X[index]]), torch.cuda.FloatTensor([self.Y[index][1]*1e9, self.Y[index][0]])

    def __len__(self):
        return len(self.X) - 100
