import numpy as np
import matplotlib.pyplot as plt
import math
import torch
from train import PerformanceMeasurementModel
from scipy.optimize import minimize_scalar
from scipy.optimize import root_scalar
import copy

size = 0


def get_NN_grad(block_size: int):
    size_scale = 1e6
    time_scale = 3e6
    mem_scale = 24
    model = PerformanceMeasurementModel(1, 2, size_scale)
    model.load_state_dict(torch.load("performanceModel_new.pth"))

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

    return time_grad.detach().numpy()[0, 0] * time_scale, memory_grad.detach().numpy()[0, 0] / mem_scale


def f_dash(x):
    t_dash, m_dash = get_NN_grad(x)
    return t_dash * (size / x) + timeOf(x) * (-1 * size / x ** 2) + m_dash


def get_prediction(block_size, model_path, size_scale, time_scale, mem_scale):
    my_model = PerformanceMeasurementModel(1, 2, size_scale)
    my_model.load_state_dict(torch.load(model_path))
    my_model = my_model.cuda()
    pred = my_model(torch.cuda.FloatTensor(np.array([[block_size]]).reshape((-1, 1))))
    pred[:, 0], pred[:, 1] = pred[:, 0] * time_scale, pred[:, 1] / mem_scale
    res = pred.detach().cpu().numpy()
    return res[0, 0], res[0, 1]


def timeOf(x):  # time of block x
    time, memory = get_prediction(x, "performanceModel_new.pth", 1e6, 3e-3, 24)
    return time


def memoryOf(x):  # size of block x
    # predict the memory needed by block x
    time, memory = get_prediction(x, "performanceModel_new.pth", 1e6, 3e-3, 24)
    return memory


def f(x):  # f(x) = time
    return timeOf(x) * math.floor(size / x) + timeOf(size % x)


def g(x):  # g(x) = memory
    return memoryOf(x)


def weighting(x):  # weighting method
    return f(x) + g(x)


def main():
    arr = [100, 200, 500, 1000, 10000, 60000, 100000, 500000, 1000000, 10000000, 1000000000]
    minimum = []
    global size
    for arr_size in arr:
        size = arr_size
        minimum.append(minimize_scalar(weighting, bounds=(1, arr_size), method='bounded').x)
    print(minimum)
    size = arr[5]
    solution = root_scalar(f_dash, method='secant', x0=minimum[5]+1000, x1=minimum[5]-10000)
    print(solution)
    plt.plot(arr, minimum)
    plt.show()


if __name__ == "__main__":
    main()
