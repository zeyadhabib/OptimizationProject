import numpy as np
from scipy.optimize import minimize
import math
import torch
from train import PerformanceMeasurementModel

size = 10000
block = 10


def get_prediction(block_size, model_path, size_scale, time_scale, mem_scale):
    my_model = PerformanceMeasurementModel(1, 2, size_scale)
    my_model.load_state_dict(torch.load(model_path))
    my_model = my_model.cuda()
    pred = my_model(torch.cuda.FloatTensor(np.array([[block_size]]).reshape((-1, 1))))
    pred[:, 0], pred[:, 1] = pred[:, 0] * time_scale, pred[:, 1] / mem_scale
    res = pred.detach().cpu().numpy()
    return res[0, 0], res[0, 1]

def timeOf(x): #time of block x
    time, memory = get_prediction(x, "performanceModel_new.pth", 50e6, 0.05, 100)
    return time

def memoryOf(x): #size of block x
    #predict the memory needed by block x
    time, memory = get_prediction(x, "performanceModel_new.pth", 50e6, 0.05, 100)
    return memory

def f(x): #f(x) = time
    return timeOf(x)*math.floor(size/x) + timeOf(size%x)

def g(x): #g(x) = memory
    return memoryOf(x)

def weighting(x): #weighting method
    return f(x) + g(x)




def main():
    minimumBlock = minimize(weighting , block , method='SLSQP')


if __name__ == "__main__":
    main()



