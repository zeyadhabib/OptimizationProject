import psutil
import os
import numpy as np
import time
import json


def writeToJSONFile(data):
    fileName = './' + 'Dataset.json'
    with open(fileName, 'w') as fp:
        json.dump(data, fp, indent=4)


# noinspection PyUnusedLocal
def measure_performance(process, array, function):
    start_mem_util = process.memory_percent()
    start_time = time.time_ns()
    array = function(array)
    end_time = time.time_ns()
    end_mem_util = process.memory_percent()
    return (end_mem_util-start_mem_util), (end_time-start_time)


def FormDataSet():
    process = psutil.Process(os.getpid())
    step = 4.9
    # data = []
    dataset_file = open("dataset.txt", "a")
    while step <= 10.0:
        step += 0.1
        size = 11
        while size <= 1e9:
            # data_set_object = {}
            array = np.random.rand(int(size))
            mem_util, exec_time = (0, 0)
            temp_mem_util, temp_exec_time = measure_performance(process, array, np.square)
            mem_util, exec_time = mem_util + temp_mem_util, exec_time + temp_exec_time
            temp_mem_util, temp_exec_time = measure_performance(process, array, np.sqrt)
            mem_util, exec_time = mem_util + temp_mem_util, exec_time + temp_exec_time
            temp_mem_util, temp_exec_time = measure_performance(process, array, np.abs)
            mem_util, exec_time = mem_util + temp_mem_util, exec_time + temp_exec_time
            temp_mem_util, temp_exec_time = measure_performance(process, array, np.sin)
            mem_util, exec_time = mem_util + temp_mem_util, exec_time + temp_exec_time
            temp_mem_util, temp_exec_time = measure_performance(process, array, np.exp)
            mem_util, exec_time = mem_util + temp_mem_util, exec_time + temp_exec_time
            mem_util, exec_time = mem_util / 5, exec_time / 5
            # data_set_object['memory'] = mem_util
            # data_set_object['time'] = exec_time
            # data_set_object['size'] = int(size)
            # print(data_set_object)
            # data.append(data_set_object)
            dataset_file.write(str(int(size)) + ' ' + str(mem_util) + ' ' + str(exec_time) + '\n')
            print(str(int(size)) + ' ' + str(mem_util) + ' ' + str(exec_time))
            size = size * step
    # writeToJSONFile(data)


def main():
    FormDataSet()


if __name__ == "__main__":
    main()
