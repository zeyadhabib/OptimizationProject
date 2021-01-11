import numpy as np
from scipy.optimize import minimize
import math
size = 100
block = []

def timeOf(x): #time of block x
    #predict the time taken by block x
    return

def sizeOf(x): #size of block x
    #predict the memory needed by block x
    return

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



