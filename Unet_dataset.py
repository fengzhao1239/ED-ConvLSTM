import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


class UDataset(Dataset):

    def __init__(self, train_state, variable_type):
        self.train_state = train_state
        self.variable_type = variable_type
        self.x_tensor_dir = 'Unet_' + str(self.train_state) + '_' + str(self.variable_type) + '_' + 'x'
        self.y_tensor_dir = 'Unet_' + str(self.train_state) + '_' + str(self.variable_type) + '_' + 'y'
        self.x_tensor_names = os.listdir(self.x_tensor_dir)
        self.y_tensor_names = os.listdir(self.y_tensor_dir)
        self.xx_list = []
        self.yy_list = []
        for xx in self.x_tensor_names:
            self.xx_list.append(torch.load(os.path.join(self.x_tensor_dir, xx)))
        for yy in self.y_tensor_names:
            self.yy_list.append(torch.load(os.path.join(self.y_tensor_dir, yy)))

    def __len__(self):
        return len(self.x_tensor_names)

    def __getitem__(self, item):
        x_tensor = self.xx_list[item]
        y_tensor = self.yy_list[item]
        return x_tensor.float(), y_tensor.float()


if __name__ == "__main__":
    datas = UDataset('val', 'p')
    dataloader = DataLoader(datas, batch_size=128, drop_last=False)
    for bb, (x, y) in enumerate(dataloader):
        print(f'batch: {bb} | x shape: {x.shape} | y shape: {y.shape}')