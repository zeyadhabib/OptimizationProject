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
    pred[:, 1].backward()
    memory_grad = copy.deepcopy(nn_input.grad)

    nn_input.grad.zero_()
    opt.zero_grad()
    pred = model(nn_input)
    pred[:, 0].backward()
    time_grad = copy.deepcopy(nn_input.grad)

    return time_grad.detach().numpy()[0, 0], memory_grad.detach().numpy()[0, 0]


def test():

    size_scale = 1e6
    time_scale = 3e6
    mem_scale = 24

    my_model = PerformanceMeasurementModel(1, 2, size_scale)
    my_model.load_state_dict(torch.load("performanceModel_new.pth"))
    X, Y = load_test_set("dataset.txt")
    x = torch.FloatTensor(X[20:21])
    x.requires_grad = True
    pred = my_model(x)
    pred[:, 0].backward()
    pred = my_model(x)
    pred[:, 1].backward()
    total = copy.deepcopy(x.grad)
    time, mem = get_NN_grad(my_model, X[20])
    print(total.detach().numpy()[0, 0] == time+mem)
    # pred[:, 0], pred[:, 1] = pred[:, 0] * time_scale, pred[:, 1] / mem_scale
    # print(torch.cat((torch.FloatTensor(X[20:30]), pred, torch.FloatTensor(Y[20:30])), dim=1))


if __name__ == '__main__':
    test()
