from abc import ABC

import torch
from JSONDataSet import JSON_DataSet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


class Linear:

    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weights = torch.randn(in_features, out_features, dtype=torch.float64, device='cuda', requires_grad=True)
        self.bias = torch.randn(size=(1, out_features), dtype=torch.float64, device='cuda', requires_grad=True)

    def forward(self, x):
        return torch.add(torch.matmul(x, self.weights), self.bias)

    def backward(self, criterion, lr):
        criterion.backward()
        with torch.no_grad():
            self.weights -= lr * self.weights.grad
            self.bias -= lr * self.bias.grad

            self.weights.grad.zero_()
            self.bias.grad.zero_()

    def __call__(self, x):
        return self.forward(x)


def train():
    train_set = JSON_DataSet("Dataset.json")
    train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)

    time_model = Linear(1, 1)
    memory_model = Linear(1, 1)
    criterion_1 = torch.nn.MSELoss()
    criterion_2 = torch.nn.MSELoss()

    avg_losses = []

    for epoch in range(1000):
        total_loss = 0
        num = 0
        for i, (size, target) in enumerate(train_loader):
            pred_y_time = time_model(torch.log(size))
            pred_y_memory = memory_model(torch.log(size))
            target[:, 0], target[:, 1] = torch.log(target[:, 0] + 1e-20), target[:, 1]
            loss_time = criterion_1(pred_y_time, target[:, 0])
            loss_memory = criterion_2(pred_y_memory, target[:, 1])
            time_model.backward(loss_time, 0.001)
            memory_model.backward(loss_memory, 0.001)
            total_loss += loss_time + loss_memory
            num += target.shape[0]
        avg_losses.append([epoch, total_loss/num])
        print('epoch {}, loss {}'.format(epoch, total_loss/num))
    avg_losses = np.array(avg_losses)
    plt.plot(avg_losses[:, 0], avg_losses[:, 1])
    plt.show()


if __name__ == '__main__':
    train()