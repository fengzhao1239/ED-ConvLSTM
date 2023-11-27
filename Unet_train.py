import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.cuda.amp import GradScaler

from Unet_model import *
from Unet_dataset import UDataset
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

dxx = [[[[0, 0, 0],
        [-1/2, 0, 1/2],
        [0, 0, 0]]]]

dyy = [[[[0, -1/2, 0],
        [0, 0, 0],
        [0, 1/2, 0]]]]

dxx_xx = [[[[0, 0, 0],
        [1, -2, 1],
        [0, 0, 0]]]]

dyy_yy = [[[[0, 1, 0],
        [0, -2, 0],
        [0, 1, 0]]]]


def loss_calculate(pred, label, loss_fn):
    pred, label = pred.squeeze(), label.squeeze()  # [b, 64, 64]
    loss_tensor = loss_fn(pred, label)
    # loss_tensor = torch.sqrt(loss_tensor+1e-12)
    return torch.mean(torch.mean(loss_tensor, dim=0))


def region_penalty(pred, label, loss_fn):
    pred, label = pred.squeeze(), label.squeeze()  # [b, 64, 64]
    pred = pred.clone()
    pred[label == 0] = pred[label == 0] * 0
    loss_tensor_1 = loss_fn(pred, label)
    # loss_tensor_1 = torch.sqrt(loss_tensor_1+1e-12)
    return torch.mean(torch.mean(loss_tensor_1, dim=0))


def metric_eval(pred, label):
    pred, label = pred.squeeze(), label.squeeze()    # [b, 64, 64]
    batch_num, _, _ = pred.size()
    r2_value = 0
    for batch in range(batch_num):
        tmp_r2 = r2_score(pred.flatten(), label.flatten())
        r2_value = r2_value + tmp_r2
    return r2_value / batch_num


def train_loop(data_loader, gpu, network, loss_fn, optim):
    network.train()
    size = len(data_loader.dataset)
    num_batch = len(data_loader)
    total_loss = 0
    total_label_loss = 0
    metric_score = 0
    print(f'num of batch = {num_batch}')

    for batch, (x, y) in enumerate(data_loader):
        x, y = x.to(gpu), y.to(gpu)
        pred = network(x.float())
        dx_filter = Dx(dxx, 1, 1).to(gpu)
        dy_filter = Dy(dyy, 1, 1).to(gpu)
        dxx_filter = Dxx(dxx_xx, 1, 1).to(gpu)
        dyy_filter = Dyy(dyy_yy, 1, 1).to(gpu)

        loss_region = region_penalty(pred, y, loss_fn)
        loss_0 = loss_calculate(pred, y, loss_fn)
        loss_1 = loss_calculate(dx_filter(pred), dx_filter(y), loss_fn) + loss_calculate(dy_filter(pred), dy_filter(y), loss_fn)
        loss_2 = loss_calculate(dxx_filter(pred), dxx_filter(y), loss_fn) + loss_calculate(dyy_filter(pred), dyy_filter(y), loss_fn)
        loss = loss_0 + 0.5 * loss_1 + 0.2 * loss_region
        metric_score += metric_eval(pred, y).item()
        loss.backward()
        optim.step()
        optim.zero_grad()

        total_loss += loss.item()
        total_label_loss += loss_0.item()
        if batch % 10 == 0:
            loss_value, label_loss_value, current_batch = loss.item(), loss_0.item(), batch * len(x)
            print(f'loss: {loss_value:>15f} | label loss: {label_loss_value:>15f}  [{current_batch:>5d}/{size:>5d}]')
    # lr_scheduler.step()
    print(
        f'## Training Error: avg loss = {total_loss / num_batch:.5e} | avg label loss = {total_label_loss / num_batch:.5e}')
    print(f'## Training R2 score: {metric_score / num_batch:.4f}')
    return total_label_loss / num_batch, metric_score / num_batch


def val_loop(data_loader, gpu, network, loss_fn):
    network.eval()

    num_batches = len(data_loader)
    val_loss = 0
    metric_score = 0

    with torch.no_grad():
        for batch, (x, y) in enumerate(data_loader):
            x, y = x.to(gpu), y.to(gpu)
            pred = network(x.float())
            loss = loss_calculate(pred, y, loss_fn)
            val_loss += loss.item()
            metric_score += metric_eval(pred, y).item()

    val_loss /= num_batches
    print(f'## Validation Error: avg loss = {val_loss:.5e}')
    print(f'## Validation R2 score: {metric_score / num_batches:.4f}')

    return val_loss, metric_score / num_batches


if __name__ == '__main__':

    now = datetime.datetime.now()
    load_start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}, GPU name: {torch.cuda.get_device_name()}')

    state_variable = 's'
    train_set = UDataset('train', state_variable)
    val_set = UDataset('val', state_variable)

    batch_size = 32

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4,
                            pin_memory=True)
    print(f'Data load finished. Takes {time.time() - load_start:.2f} seconds.')

    model = UNET(in_channels=4, out_channels=1).float().to(device)

    learning_rate = 5.e-4
    criterion = nn.MSELoss(reduction='none')  # L1 loss for pressure, L2 loss for saturation
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # scheduler = MultiStepLR(adam_optimizer, milestones=[40, 90, 150], gamma=0.5)
    # scheduler = StepLR(adam_optimizer, step_size=40, gamma=0.6)
    scheduler = ReduceLROnPlateau(adam_optimizer, mode='min', factor=0.5, patience=8, verbose=False, threshold=0.0001,
                                  threshold_mode='rel', cooldown=0, min_lr=1.e-8, eps=1e-08)

    epochs = 100

    begins = time.time()
    train_loss_list = []
    test_loss_list = []
    train_metric_list = []
    test_metric_list = []
    for tt in range(epochs):
        print(f'Epoch {tt + 1}\n----------------------------------')
        begin1 = time.time()
        train_loss_value, train_metric = train_loop(train_loader, device, model, criterion, adam_optimizer)
        test_loss_value, test_metric = val_loop(val_loader, device, model, criterion)
        scheduler.step(test_loss_value)
        train_loss_list.append(train_loss_value)
        test_loss_list.append(test_loss_value)
        train_metric_list.append(train_metric)
        test_metric_list.append(test_metric)
        end1 = time.time()
        # if early_stopper.early_step(test_loss_value):
        #     break
        print(f'current learning rate: {adam_optimizer.param_groups[0]["lr"]}')
        print(f'******* This epoch costs {(end1 - begin1) / 60:.2f} min. *******')
        print(f'******* All epochs costs {(end1 - begins) / 60:.2f} min. *******\n\n')
    print('=========Done===========')

    ends = time.time()
    print(f'Total time is {(ends - begins) / 60.:.2f} mins and {(ends - begins) / 3600.:.2f} hours.')

    test_num = 1

    torch.save(model.state_dict(), f'Unet_model_weights_10.12_{state_variable}_test3.pth')
    np.savetxt(f'Unet_training_loss_{state_variable}_test3.txt', train_loss_list)
    np.savetxt(f'Unet_testing_loss_{state_variable}_test3.txt', test_loss_list)
    np.savetxt(f'Unet_training_metric_{state_variable}_test3.txt', train_metric_list)
    np.savetxt(f'Unet_testing_metric_{state_variable}_test3.txt', test_metric_list)

    plt.semilogy(np.arange(len(train_loss_list)), train_loss_list, c='r', marker='o', label='training loss')
    plt.semilogy(np.arange(len(test_loss_list)), test_loss_list, c='b', marker='o', label='validating loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'Unet_Loss Values_10.12_{state_variable}_test3.png', dpi=300)
    plt.close()

    plt.plot(np.arange(len(train_metric_list))[1:], train_metric_list[1:], c='r', marker='x', label='training R2')
    plt.plot(np.arange(len(test_metric_list))[1:], test_metric_list[1:], c='b', marker='x', label='validating R2')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'Unet_R2 Values_10.12_{state_variable}_test3.png', dpi=300)
    plt.close()