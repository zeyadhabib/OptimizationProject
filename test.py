import torch
from torch import nn
from train import PerformanceMeasurementModel
import numpy as np
import copy


def load_test_set(path):
    f = open(path, 'r')
    lines = f.readlines()
    X, Y = [], []
    for line in lines:
        elements = line.split(' ')
        X.append(int(elements[0]))
        Y.append([float(elements[2]), float(elements[1])])
    return np.array(X).reshape(-1, 1), np.array(Y)


def get_NN_grad(model: nn.Module, block_size: float):

    nn_input = torch.FloatTensor(np.array([block_size]).reshape((-1, 1)))
    nn_input.requires_grad = True
    opt = torch.optim.Adam(params=model.parameters())

    pred = model(nn_input)
    dummy_memory = torch.zeros_like(pred)
    dummy_memory[:, 0] = pred[:, 0]
    pred2 = torch.nn.MSELoss()(pred, dummy_memory)
    pred2.backward()
    memory_grad = copy.deepcopy(nn_input.grad)

    nn_input.grad.zero_()
    opt.zero_grad()

    pred = model(nn_input)
    dummy_time = torch.zeros_like(pred)
    dummy_time[:, 1] = pred[:, 1]
    pred1 = torch.nn.MSELoss()(pred, dummy_time)
    pred1.backward()
    time_grad = copy.deepcopy(nn_input.grad)

    return time_grad.detach().numpy()[0, 0], memory_grad.detach().numpy()[0, 0]


def test():

    size_scale = 1e6
    time_scale = 3e6
    mem_scale = 24

    my_model = PerformanceMeasurementModel(1, 2, size_scale)
    my_model.load_state_dict(torch.load("performanceModel_new.pth"))
    X, Y = load_test_set("dataset.txt")
    pred = my_model(torch.FloatTensor(X[20:30]))
    pred[:, 0], pred[:, 1] = pred[:, 0] * time_scale, pred[:, 1] / mem_scale
    print(torch.cat((torch.FloatTensor(X[20:30]), pred, torch.FloatTensor(Y[20:30])), dim=1))


if __name__ == '__main__':
    test()
