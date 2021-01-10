import torch
from torch.utils.data import Dataset
import json


class JSON_DataSet(Dataset):

    @staticmethod
    def readFromJSONFile(fileName):
        dataset = json.load(open(fileName))
        return dataset

    def __init__(self, json_file_path):
        super(JSON_DataSet, self).__init__()
        self.json_file_path = json_file_path
        self.raw_data = JSON_DataSet.readFromJSONFile(json_file_path)

    def __getitem__(self, index):
        return torch.cuda.FloatTensor([int(self.raw_data[index]['size'])]),\
               torch.cuda.FloatTensor([float(self.raw_data[index]['time']), float(self.raw_data[index]['memory'])])

    def __len__(self):
        return len(self.raw_data)
