import torch
from train import PerformanceMeasurementModel
import numpy as np


def load_test_set(path):
    f = open(path, 'r')
    lines = f.readlines()
    X, Y = [], []
    for line in lines:
        elements = line.split(' ')
        X.append(int(elements[0]))
        Y.append([float(elements[2]), float(elements[1])])
    return np.array(X).reshape(-1, 1), np.array(Y)


def test():
    size_scale = 1e5
    time_scale = 5e5

    # size_scale = 50e6
    # time_scale = 50e6

    mem_scale = 100

    my_model = PerformanceMeasurementModel(1, 2, size_scale)
    my_model.load_state_dict(torch.load("performanceModel_new.pth"))
    X, Y = load_test_set("dataset.txt")
    pred = my_model(torch.FloatTensor(X[10:20]))
    pred[:, 0], pred[:, 1] = pred[:, 0] * time_scale, pred[:, 1] / mem_scale
    print(torch.cat((torch.FloatTensor(X[10:20]), pred, torch.FloatTensor(Y[10:20])), dim=1))


if __name__ == '__main__':
    test()
