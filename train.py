from abc import ABC

import copy
import torch
from TextDataSet import TextDataSet, get_train_val_samplers
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


class PerformanceMeasurementModel(torch.nn.Module, ABC):

    def __init__(self, in_features, out_features, size_scale):
        super(PerformanceMeasurementModel, self).__init__()
        self.size_scale = size_scale
        self.linear_1 = torch.nn.Linear(in_features, 32)
        self.linear_4 = torch.nn.Linear(32, out_features, bias=False)

    def forward(self, x):
        x_1 = torch.nn.functional.relu(self.linear_1(x.contiguous() / self.size_scale))
        x_1 = torch.nn.functional.relu(self.linear_4(x_1))
        return x_1


def train():
    size_scale = 1e6
    time_scale = 3e6
    mem_scale = 24
    validation_split = 0.1

    train_set = TextDataSet("dataset_final.txt")
    train_sampler, valid_sampler = get_train_val_samplers(train_set, validation_split)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=256, sampler=train_sampler
    )

    valid_loader = DataLoader(
        dataset=train_set,
        batch_size=64, sampler=valid_sampler
    )

    our_model = PerformanceMeasurementModel(1, 2, size_scale)
    our_model = our_model.cuda()

    optimizer = torch.optim.Adam(our_model.parameters(), lr=0.01, weight_decay=0.001)
    criterion = torch.nn.MSELoss()
    metric = torch.nn.L1Loss()

    avg_losses = []
    avg_losses_valid = []
    state_dict = None
    prev_loss = 1e11

    for epoch in range(30):
        total_loss, total_loss_valid, total_loss_metric = (0, 0, 0)
        num, num_valid = (0, 0)
        for i, ((size, target), (size_valid, target_valid)) in enumerate(zip(train_loader, valid_loader)):
            pred_y = our_model(size)
            target[:, 0], target[:, 1] = target[:, 0] / time_scale, target[:, 1] * mem_scale

            loss = criterion(pred_y, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss * target.shape[0]
            num += target.shape[0]

            with torch.no_grad():
                pred_y_valid = our_model(size_valid)

                pred_y_metric = copy.deepcopy(pred_y_valid)
                target_valid_metric = copy.deepcopy(target_valid)

                target_valid[:, 0], target_valid[:, 1] = target_valid[:, 0] / time_scale, target_valid[:, 1] * mem_scale
                loss_valid = criterion(pred_y_valid, target_valid)
                total_loss_valid += loss_valid * target_valid.shape[0]
                num_valid += target_valid.shape[0]

                pred_y_metric[:, 0], pred_y_metric[:, 1] = pred_y_metric[:, 0] * time_scale, \
                                                           pred_y_metric[:, 1] / mem_scale
                loss_metric = metric(pred_y_metric, target_valid_metric)

                total_loss_metric += loss_metric * target_valid_metric.shape[0]

        print('epoch {}, loss {}, validation loss {}, metric {}'.format(epoch, total_loss / num,
                                                                        total_loss_valid / num_valid,
                                                                        total_loss_metric / num_valid))
        avg_losses.append(total_loss / num)
        avg_losses_valid.append(total_loss_valid / num_valid)

        if total_loss / num < prev_loss:
            print("********************SAVING*******************")
            state_dict = copy.deepcopy(our_model.state_dict())
            prev_loss = total_loss / num

    avg_losses = np.array(avg_losses)
    avg_losses_valid = np.array(avg_losses_valid)
    plt.plot(range(len(avg_losses)), avg_losses)
    plt.plot(range(len(avg_losses_valid)), avg_losses_valid)
    torch.save(state_dict, "performanceModel_new.pth")

    plt.show()


if __name__ == '__main__':
    train()
