import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


class TextDataSet(Dataset):

    def __init__(self, text_file_path):
        super(TextDataSet, self).__init__()
        self.json_file_path = text_file_path
        f = open(text_file_path, 'r')
        self.raw_data = f.readlines()

    def __getitem__(self, index):
        line = self.raw_data[index].split(' ')
        return torch.cuda.FloatTensor([int(line[0])]), torch.cuda.FloatTensor([float(line[2]), float(line[1])])

    def __len__(self):
        return len(self.raw_data)


def get_train_val_samplers(dataset, validation_split):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler
