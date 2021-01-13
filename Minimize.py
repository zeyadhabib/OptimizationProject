import numpy as np
import matplotlib.pyplot as plt
import math
import torch
from train import PerformanceMeasurementModel
from scipy.optimize import minimize_scalar
from scipy.optimize import fsolve


size=0


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

    return time_grad.detach().numpy()[0, 0] * time_scale, memory_grad.detach().numpy()[0, 0] / mem_scale

def f_dash(x):
  t_dash,m_dash = get_NN_grad(x)
  return t_dash * (size/x) + timeOf(x) * (-1*size/x**2) + m_dash

def get_prediction(block_size, model_path, size_scale, time_scale, mem_scale):
    my_model = PerformanceMeasurementModel(1, 2, size_scale)
    my_model.load_state_dict(torch.load(model_path))
    my_model = my_model.cuda()
    pred = my_model(torch.cuda.FloatTensor(np.array([[block_size]]).reshape((-1, 1))))
    pred[:, 0], pred[:, 1] = pred[:, 0] * time_scale, pred[:, 1] / mem_scale
    res = pred.detach().cpu().numpy()
    return res[0, 0], res[0, 1]

def timeOf(x): #time of block x
    time, memory = get_prediction(x, "performanceModel_new.pth", 1e6, 3e-3, 24)
    return time

def memoryOf(x): #size of block x
    #predict the memory needed by block x
    time,memory = get_prediction(x, "performanceModel_new.pth", 1e6, 3e-3, 24)
    return memory

def f(x): #f(x) = time
    return timeOf(x)*math.floor(size/x) + timeOf(size%x)

def g(x): #g(x) = memory
    return memoryOf(x)

def weighting(x): #weighting method
    return f(x) + g(x)




def main():
    arr = [100, 200, 500, 1000, 10000, 60000, 100000, 500000, 1000000, 10000000, 1000000000]
    minimum = []
    for i in range(len(arr)):
        size = arr[i]
        minimum.append(minimize_scalar(weighting, bounds=(1, size), method='bounded').x)
    print(minimum)
    plt.plot(arr, minimum)
    solution = fsolve(f_dash, minimum[4])
    print(solution)


if __name__ == "__main__":
    main()



