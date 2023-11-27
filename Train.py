"""
Training script for ED-ConvLSTM
"""

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
# from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from Model_simple import *
from MyDataset import MyDataset
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau
from torcheval.metrics.functional import r2_score
import time
import datetime
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.autograd.set_detect_anomaly(False)
torch.manual_seed(3407)
torch.set_default_dtype(torch.float32)
ts = 24    # using only 24 months data for training

# finite difference filters, for calculating 1st & 2nd derivatives
# 1 st derivative of x and y axes
dxx = [[[[0, 0, 0],
        [-1/2, 0, 1/2],
        [0, 0, 0]]]] * ts

dyy = [[[[0, -1/2, 0],
        [0, 0, 0],
        [0, 1/2, 0]]]] * ts

# 2 nd derivative of x and y axes
dxx_xx = [[[[0, 0, 0],
        [1, -2, 1],
        [0, 0, 0]]]] * ts

dyy_yy = [[[[0, 1, 0],
        [0, -2, 0],
        [0, 1, 0]]]] * ts


# below 3 functions are used for calculating loss
def loss_through_time(pred, label, loss_fn):
    """
    for backpropagation through time, we need to calculate loss through time.
    note that pressure and saturation are predicted separately.
    :param pred: predictions from neural network with shape (batch_size, 24, 1, 64, 64)
    :param label: labels with shape (batch_size, 24, 1, 64, 64)
    :param loss_fn: the loss function to be calculated (L1 loss for pressure, MSE loss for saturation)
    :return: the mean loss value of the current batch
    """
    pred, label = pred.squeeze(), label.squeeze()    # [batch, 24, 64, 64]
    loss_tensor = loss_fn(pred, label)    # calculate loss
    sum_time = torch.sum(loss_tensor, dim=1) + torch.sum(loss_tensor[:, :10, :, :], dim=1) * 5.0 + loss_tensor[:, 0, :, :] * 10    # temporal penalty, coefficients are hyperparameters to be tuned
    mean_batch = torch.mean(sum_time, dim=0)    # add along the time axis
    return torch.mean(mean_batch)


def grad_loss(pred, label, loss_fn):
    """
    this function is for derivative loss
    """
    pred, label = pred.squeeze(), label.squeeze()  # [batch, 24, 64, 64]
    loss_tensor = loss_fn(pred, label)
    sum_time = torch.sum(loss_tensor, dim=1)
    mean_batch = torch.mean(sum_time, dim=0)
    return torch.mean(mean_batch)


def region_penalty(pred, label, loss_fn):
    """
    this function is for regional penalty especially for saturation
    """
    pred, label = pred.squeeze(), label.squeeze()
    pred = pred.clone()
    pred[label == 0] = pred[label == 0] * 0    # mask out saturation pixels with 0 values
    loss_tensor = loss_fn(pred, label)
    sum_time = torch.sum(loss_tensor, dim=1) + torch.sum(loss_tensor[:, :10, :, :], dim=1) * 5.0 + loss_tensor[:, 0, :, :] * 10    # temporal penalty as above
    mean_batch = torch.mean(sum_time, dim=0)
    return torch.mean(mean_batch)


def metric_eval(pred, label):
    """
    this function is only used to validate the model performance during training
    here r2 score is used as the evaluation metric
    """
    pred, label = pred.squeeze(), label.squeeze()    # [batch, 24, 64, 64]
    batch_num, seq_length, _, _ = pred.size()
    r2_value = 0
    for batch in range(batch_num):
        batch_r2 = 0
        for time_snap in range(seq_length):
            tmp_pred, tmp_label = pred[batch, time_snap, :, :], label[batch, time_snap, :, :]    # [64, 64]
            tmp_r2 = r2_score(tmp_pred.flatten(), tmp_label.flatten())
            batch_r2 = batch_r2 + tmp_r2
        batch_r2 = batch_r2 / seq_length
        r2_value = r2_value + batch_r2
    return r2_value / batch_num


def train_loop(dataloader, gpu, network, dx_filter, dy_filter, dxx_filter, dyy_filter, loss_fn, optimizer):
    """
    the training function
    :param dataloader: Dataloader
    :param gpu: the training device
    :param network: the neural network
    :param dx_filter: non-trainable CNN layer for calculating dx derivative
    :param dy_filter: non-trainable CNN layer for calculating dy derivative
    :param dxx_filter: non-trainable CNN layer for calculating d2x derivative
    :param dyy_filter: non-trainable CNN layer for calculating d2y derivative
    :param loss_fn: the loss function
    :param optimizer: Optimizer
    :return: loss value and r2 score
    """
    network.train()
    size = len(dataloader.dataset)
    num_batch = len(dataloader)
    total_loss = 0    # loss values including all the items
    total_label_loss = 0    # loss values only for the state variables
    metric_score = 0
    print(f'num of batch = {num_batch}')

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(gpu), y.to(gpu)
        pred, pred_dx, pred_dy, pred_dxx, pred_dyy, hidden_states = network(x)
        loss_0 = loss_through_time(pred, y, loss_fn)
        loss_dx = grad_loss(pred_dx, dx_filter(y.squeeze()), loss_fn)
        loss_dy = grad_loss(pred_dy, dy_filter(y.squeeze()), loss_fn)
        loss_dxx = grad_loss(pred_dxx, dxx_filter(y.squeeze()), loss_fn)
        loss_dyy = grad_loss(pred_dyy, dyy_filter(y.squeeze()), loss_fn)
        loss_region = region_penalty(pred, y, loss_fn)
        # coefficients: pressure 0.8 0 0 L1; saturation 0.8 0.5 0.5 L2
        loss = loss_0 + (loss_dx + loss_dy) * 0.8 + (loss_dxx + loss_dyy) * 0.5 + loss_region * 0.5
        metric_score += metric_eval(pred, y).item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        total_label_loss += loss_0.item()

        if batch % 50 == 0:
            loss_value, label_loss_value, current_batch = loss.item(), loss_0.item(), batch * len(x)
            print(f'loss: {loss_value:>15f} | label loss: {label_loss_value:>15f}  [{current_batch:>5d}/{size:>5d}]')
    print(f'## Training Error: avg loss = {total_loss / num_batch:.5e} | avg label loss = {total_label_loss / num_batch:.5e}, Training R2 score: {metric_score / num_batch:.4f}')
    return total_label_loss / num_batch, metric_score / num_batch


def val_loop(dataloader, gpu, network, loss_fn):
    """
    the validation function
    :param dataloader: Dataloader
    :param gpu: the validating device
    :param network: the neural network
    :param loss_fn: the loss function
    :return: validation loss and r2 score on validation set
    """
    network.eval()

    num_batches = len(dataloader)
    val_loss = 0
    metric_score = 0

    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(gpu), y.to(gpu)
            pred, _ = network(x, )
            loss = grad_loss(pred, y, loss_fn)
            val_loss += loss.item()
            metric_score += metric_eval(pred, y).item()

    val_loss /= num_batches
    print(f'&& Validation Error: avg loss = {val_loss:.5e}, Validation R2 score: {metric_score / num_batches:.4f}')

    return val_loss, metric_score / num_batches


if __name__ == '__main__':

    now = datetime.datetime.now()
    load_start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}, GPU name: {torch.cuda.get_device_name()}')

    os.mkdir(f'log_directory\\{now.strftime("%Y%m%d-%H%M%S")}')    # use tensorboard for real-time monitoring
    writer = SummaryWriter(f'log_directory\\{now.strftime("%Y%m%d-%H%M%S")}')

    state_variable = 's'    # s for saturation, p for pressure

    # Dataset initialization
    train_set = MyDataset('train', state_variable)
    val_set = MyDataset('val', state_variable)

    batch_size = 32

    # Dataloader initialization
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4,
                            pin_memory=True)
    print(f'Data load finished. Takes {time.time() - load_start:.2f} seconds.')

    model = InceptionLSTM(3, 1, 128, ts).float().to(device)    # 3 for input channel, 1 for output channel, 128 for hidden size, ts=24 for time steps used for training

    # We use CNN models with one layer of non-trainable conv filter to calculate derivatives
    model_dx = Dx(dxx, ts, ts).float().to(device)
    model_dy = Dy(dyy, ts, ts).float().to(device)
    model_dxx = Dxx(dxx_xx, ts, ts).float().to(device)
    model_dyy = Dyy(dyy_yy, ts, ts).float().to(device)

    learning_rate = 5.e-4
    criterion = nn.MSELoss(reduction='none')    # L1 loss for pressure, L2 loss for saturation
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    scheduler = ReduceLROnPlateau(adam_optimizer, mode='min', factor=0.2, patience=10, verbose=False, threshold=0.0001,
                      threshold_mode='rel', cooldown=0, min_lr=1.e-7, eps=1e-08)    # automatic learning rate decay

    epochs = 100

    # some recordings
    begins = time.time()
    train_loss_list = []
    val_loss_list = []
    train_metric_list = []
    val_metric_list = []

    for tt in range(epochs):
        print(f'Epoch {tt + 1}\n----------------------------------')
        begin1 = time.time()

        train_loss_value, train_metric = train_loop(train_loader, device, model, model_dx, model_dy, model_dxx, model_dyy, criterion, adam_optimizer)    # training loop
        val_loss_value, val_metric = val_loop(val_loader, device, model, criterion)    # validation loop

        scheduler.step(val_loss_value)    # automatically reduce learning rate based on the performance on validation set

        # write records to tensorboard:
        writer.add_scalar('Loss/train', train_loss_value, global_step=tt+1)
        writer.add_scalar('Loss/validation', val_loss_value, global_step=tt+1)
        writer.add_scalar('R2 on Validation', val_metric, global_step=tt+1)

        # save loss values and r2 score
        train_loss_list.append(train_loss_value)
        val_loss_list.append(val_loss_value)
        train_metric_list.append(train_metric)
        val_metric_list.append(val_metric)
        end1 = time.time()

        print(f'current learning rate: {adam_optimizer.param_groups[0]["lr"]}')
        print(f'******* This epoch costs {(end1 - begin1) / 60:.2f} min. *******')
        print(f'******* All epochs costs {(end1 - begins) / 60:.2f} min. *******\n\n')
    print('=========Done===========')

    ends = time.time()
    writer.close()
    print(f'Total time is {(ends - begins) / 60.:.2f} mins and {(ends - begins) / 3600.:.2f} hours.')

    torch.save(model.state_dict(), f'model_weights_{now.strftime("%m.%d-%H.%M")}_{state_variable}.pth')
    np.savetxt(f'training_loss_{state_variable}_{now.strftime("%m.%d-%H.%M")}.txt', train_loss_list)
    np.savetxt(f'testing_loss_{state_variable}_{now.strftime("%m.%d-%H.%M")}.txt', val_loss_list)
    np.savetxt(f'training_metric_{state_variable}_{now.strftime("%m.%d-%H.%M")}.txt', train_metric_list)
    np.savetxt(f'testing_metric_{state_variable}_{now.strftime("%m.%d-%H.%M")}.txt', val_metric_list)