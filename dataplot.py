import matplotlib.pyplot as plt
import numpy as np

file = open("dataset_final.txt", "r")
X, Y = [], []
lines = file.readlines()

for line in lines:
    elements = line.split(" ")
    X.append(int(elements[0]))
    Y.append([float(elements[1]), float(elements[2])])

X = np.array(X)
Y = np.array(Y)
# X = np.load("X.npy")
# Y = np.load("Y.npy")
idx = np.argsort(X)
X = X[idx]
Y = Y[idx]
plt.figure()
plt.plot(X/1e5, Y[:, 0]*100)
# plt.figure()
plt.plot(X/1e5, Y[:, 1]/5e5)
plt.show()
