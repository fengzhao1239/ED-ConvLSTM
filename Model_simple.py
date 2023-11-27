"""
The Neural Network: ED-ConvLSTM
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import drop_block2d
import torch.nn.functional as F
import torch.nn.init as init
import random


torch.set_default_dtype(torch.float32)

# finite difference filters, for calculating 1st & 2nd derivatives
# 1 st derivative of x and y axes
dx = [[[[0, 0, 0],
        [-1 / 2, 0, 1 / 2],
        [0, 0, 0]]]]

dy = [[[[0, -1 / 2, 0],
        [0, 0, 0],
        [0, 1 / 2, 0]]]]

# 2 nd derivative of x and y axes
dxx = [[[[0, 0, 0],
         [1, -2, 1],
         [0, 0, 0]]]]

dyy = [[[[0, 1, 0],
         [0, -2, 0],
         [0, 1, 0]]]]


# -------------------------------- ConvLSTM cell --------------------------------
class ConvLSTMCell(nn.Module):
    """
    in_channels: input channels
    hidden_channels: hidden channels
    kernel_size: same to all convolution operations = 3
    stride: step 1
    padding: same padding
    """

    def __init__(self, in_channels, hidden_channels,
                 kernel_size=3, stride=1, padding='same'):
        super(ConvLSTMCell, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        # convolution operations
        self.Wix = nn.Sequential(nn.Conv2d(self.in_channels, self.hidden_channels,
                                           self.kernel_size, self.stride, self.padding, bias=False),
                                 nn.BatchNorm2d(self.hidden_channels))
        self.Wih = nn.Sequential(nn.Conv2d(self.hidden_channels, self.hidden_channels,
                                           self.kernel_size, self.stride, self.padding, bias=False),
                                 nn.BatchNorm2d(self.hidden_channels))
        self.Wfx = nn.Sequential(nn.Conv2d(self.in_channels, self.hidden_channels,
                                           self.kernel_size, self.stride, self.padding, bias=False),
                                 nn.BatchNorm2d(self.hidden_channels))
        self.Wfh = nn.Sequential(nn.Conv2d(self.hidden_channels, self.hidden_channels,
                                           self.kernel_size, self.stride, self.padding, bias=False),
                                 nn.BatchNorm2d(self.hidden_channels))
        self.Wcx = nn.Sequential(nn.Conv2d(self.in_channels, self.hidden_channels,
                                           self.kernel_size, self.stride, self.padding, bias=False),
                                 nn.BatchNorm2d(self.hidden_channels))
        self.Wch = nn.Sequential(nn.Conv2d(self.hidden_channels, self.hidden_channels,
                                           self.kernel_size, self.stride, self.padding, bias=False),
                                 nn.BatchNorm2d(self.hidden_channels))
        self.Wox = nn.Sequential(nn.Conv2d(self.in_channels, self.hidden_channels,
                                           self.kernel_size, self.stride, self.padding, bias=False),
                                 nn.BatchNorm2d(self.hidden_channels))
        self.Woh = nn.Sequential(nn.Conv2d(self.hidden_channels, self.hidden_channels,
                                           self.kernel_size, self.stride, self.padding, bias=False),
                                 nn.BatchNorm2d(self.hidden_channels))

    def forward(self, x, h, c):
        # print(f'x shape {x.shape}, h shape {h.shape}, c shape {c.shape}')
        i_t = torch.sigmoid(self.Wix(x) + self.Wih(h))
        f_t = torch.sigmoid(self.Wfx(x) + self.Wfh(h))
        c_t_1 = torch.tanh(self.Wcx(x) + self.Wch(h))
        o_t = torch.sigmoid(self.Wox(x) + self.Woh(h))
        c_t = f_t * c + i_t * c_t_1
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


# -------------------------------- channel attention --------------------------------
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    """
    gate_channels: num of neurons of first layer of mlp
    reduction_ratio: ratio between first and second layer of mlp
    pool_types: avg and max pooling
    return the channel refined features
    """
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        # do maxpooling and avgpooling to get channel attention weights
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw    # add both

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)    # get the channel weights
        return x * scale


# -------------------------------- spatial attention --------------------------------
class BasicConv(nn.Module):
    """
    basic convolution layer with batchnormalization and relu activation
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.LeakyReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)    # cat max and mean pooling


class SpatialGate(nn.Module):
    """
    return the spatial refined features
    """
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)    # cat max and mean pooling
        x_out = self.spatial(x_compress)    # basic conv layer
        scale = F.sigmoid(x_out)  # get the spatial weights
        return x * scale


# -------------------------------- attention block --------------------------------
class CBAM(nn.Module):
    """
    assemble the CBAM module
    """
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


# -------------------------------- 3 layers of conv --------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same', padding_mode='zeros',
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same', padding_mode='zeros',
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same', padding_mode='zeros',
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

        self.conv_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.attention_block = CBAM(out_channels, reduction_ratio=8, no_spatial=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.attention_block(x)
        return x


# -------------------------------- Inception residual block --------------------------------
# for details about forward, refer to the paper
class InceptionResNetA(nn.Module):
    def __init__(self, in_channels, out_channels, final_out_channels):
        super(InceptionResNetA, self).__init__()
        self.branch_0 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
        )
        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
            nn.Conv2d(out_channels, int(out_channels * 1.5), 3, stride=1, padding=1, bias=False),
            nn.Conv2d(int(out_channels * 1.5), int(out_channels * 2), 3, stride=1, padding=1, bias=False),
        )
        self.conv = nn.Conv2d(out_channels * 4, final_out_channels, 1, stride=1, padding=0, bias=True)
        self.main = nn.Conv2d(in_channels, final_out_channels, 1, stride=1, padding=0, bias=True)
        self.relu = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(int(out_channels * 2))
        self.attention_block = CBAM(final_out_channels, reduction_ratio=8, no_spatial=False)    # enhanced with CBAM

    def forward(self, x):
        x0 = self.branch_0(x)
        x0 = self.bn1(x0)
        x1 = self.branch_1(x)
        x1 = self.bn2(x1)
        x2 = self.branch_2(x)
        x2 = self.bn3(x2)
        x_res = torch.cat((x0, x1, x2), dim=1)
        x_res = self.conv(x_res)
        x_main = self.main(x)
        out = self.relu(x_main + x_res)
        final_out = self.attention_block(out) + out
        return final_out


# -------------------------------- Inception reduction block --------------------------------
# for details about forward, refer to the paper
class ReductionA(nn.Module):
    def __init__(self, in_channels):
        super(ReductionA, self).__init__()
        # after each branch, the feature shrinks to its 1/2 size
        self.branch_0 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.branch_1 = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1, bias=False)
        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False),
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x1 = self.bn2(x1)
        x2 = self.branch_2(x)
        out = (x0 + x1 + x2)    # use sum instead of concatenate
        return out


# -------------------------------- Encoder --------------------------------
class InceptionEncoder(nn.Module):
    def __init__(self, input_features):
        super(InceptionEncoder, self).__init__()
        self.layer_0 = InceptionResNetA(input_features, input_features, 16)
        self.layer_1 = ReductionA(16)
        self.layer_2 = InceptionResNetA(16, 16, 64)
        self.layer_3 = ReductionA(64)
        self.layer_4 = nn.Sequential(InceptionResNetA(64, 64, 128),
                                     CBAM(128, reduction_ratio=16),
                                     )

    def forward(self, x):
        # also return x32 and x64 for skip connection
        # x16 is the tensor to enter LSTM
        x64_0 = self.layer_0(x)
        x32_0 = self.layer_1(x64_0)
        x32_1 = self.layer_2(x32_0)
        x16_0 = self.layer_3(x32_1)
        x16_1 = self.layer_4(x16_0)
        return x16_1, x32_1, x64_0


# -------------------------------- Decoder --------------------------------
class InceptionDecoder(nn.Module):
    def __init__(self, out_channels):
        super(InceptionDecoder, self).__init__()
        self.decode_1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.decode_2 = ConvBlock(64+128, 64)
        self.decode_3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.decode_4 = ConvBlock(16+64, 32)
        self.conv1x1 = nn.Conv2d(32, out_channels, 3, stride=1, padding=1, bias=True)
        init.zeros_(self.conv1x1.bias)

    def forward(self, x16, x32, x64):
        # here note the skip connection from Encoder
        x1 = self.decode_1(x16)
        x2 = self.decode_2(torch.cat((x1, x32), dim=1))
        x3 = self.decode_3(x2)
        x4 = self.decode_4(torch.cat((x3, x64), dim=1))
        x6 = self.conv1x1(x4)
        return x6


# -----------------below are non-trainable CNNs for calculating derivatives-----------------------
class Dx(nn.Module):
    def __init__(self, dx_filter, in_channel, out_channel):
        super(Dx, self).__init__()
        self.conv_dx = nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='replicate', groups=in_channel,
                                 bias=False)
        self.conv_dx.weight = nn.Parameter(torch.FloatTensor(dx_filter), requires_grad=False)

    def forward(self, x):
        dx_value = self.conv_dx(x)
        return dx_value


class Dxx(nn.Module):
    def __init__(self, dx_filter, in_channel, out_channel):
        super(Dxx, self).__init__()
        self.conv_dx = nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='replicate', groups=in_channel,
                                 bias=False)
        self.conv_dx.weight = nn.Parameter(torch.FloatTensor(dx_filter), requires_grad=False)

    def forward(self, x):
        dx_value = self.conv_dx(x)
        # dxx_value = self.conv_dx(dx_value)
        return dx_value


class Dy(nn.Module):
    def __init__(self, dy_filter, in_channel, out_channel):
        super(Dy, self).__init__()
        self.conv_dy = nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='replicate', groups=in_channel,
                                 bias=False)
        self.conv_dy.weight = nn.Parameter(torch.FloatTensor(dy_filter), requires_grad=False)

    def forward(self, x):
        dy_value = self.conv_dy(x)
        return dy_value


class Dyy(nn.Module):
    def __init__(self, dy_filter, in_channel, out_channel):
        super(Dyy, self).__init__()
        self.conv_dy = nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='replicate', groups=in_channel,
                                 bias=False)
        self.conv_dy.weight = nn.Parameter(torch.FloatTensor(dy_filter), requires_grad=False)

    def forward(self, x):
        dy_value = self.conv_dy(x)
        # dyy_value = self.conv_dy(dy_value)
        return dy_value


# -------------------------------- Final architecture --------------------------------
class InceptionLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, lstm_hidden, num_steps):
        """
        the final ED-ConvLSTM
        :param in_channels: input feature channel = 3: permeability, porosity, injection rates * well location
        :param out_channels: output channel = 1: pressure or saturation
        :param lstm_hidden: num of LSTM hidden size
        :param num_steps: 24 time steps used here
        """
        super(InceptionLSTM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lstm_hidden = lstm_hidden
        self.num_steps = num_steps
        self.inception_encode = InceptionEncoder(self.in_channels)
        self.conv_lstm = ConvLSTMCell(self.lstm_hidden, self.lstm_hidden)
        self.decoder = InceptionDecoder(self.out_channels)

        self.dx = Dx(dx, 1, 1)
        self.dy = Dy(dy, 1, 1)
        self.dxx = Dxx(dxx, 1, 1)
        self.dyy = Dyy(dyy, 1, 1)

        # initialization of the first hidden state and cell memory of ConvLSTM: zero initialization
        # can be trainable parameters as well, or other initialization methods
        self.h0 = nn.Parameter(torch.zeros(1, self.lstm_hidden, 16, 16)*0.1, requires_grad=False).cuda()
        self.c0 = nn.Parameter(torch.zeros(1, self.lstm_hidden, 16, 16)*0.1, requires_grad=False).cuda()

    def forward(self, x):
        """
        x: [batch, time_steps, in_channels(5 features + 1 state variable), 64, 64]
        y: [batch, time_steps, out_channels, 64, 64]
        --------------------------------------------------
        Note:
        x has channels of 6
        1) permeability
        2) porosity
        3) anisotropy angle -- not considered as input anymore
        4) well location (binary image)  -- combined with 5)
        5) injection rates (image with a constant)
        6) true state variable of previous time step
        so the final input channel to the network is 3
        """
        hidden_list = []    # hidden state
        out_list = []    # output
        # derivatives list:
        dx_list = []
        dy_list = []
        dxx_list = []
        dyy_list = []

        batch_size, time_steps, in_features, H, W = x.size()
        h0 = self.h0.expand(batch_size, -1, -1, -1)
        c0 = self.c0.expand(batch_size, -1, -1, -1)

        hidden_state = (h0, c0)
        hidden_list.append(hidden_state)

        if self.training:
            for step in range(self.num_steps):
                # first we select features as mentioned above in the NOTE
                x_input = x[:, step, [0, 1], :, :]
                well_loc, well_para = x[:, step, 3, :, :], x[:, step, 4, :, :]
                well = well_loc * well_para
                well = well.unsqueeze(1)
                x_input = torch.cat((x_input, well), dim=1)

                encoded_x, x32, x64 = self.inception_encode(x_input)  # [batch, 3, 64, 64]
                curr_h, curr_c = self.conv_lstm(encoded_x, hidden_list[step][0], hidden_list[step][1])
                hidden_list.append((curr_h, curr_c))
                final_out = self.decoder(curr_h, x32, x64)  # [batch, 1, 64, 64]

                # derivatives
                final_out_dx = self.dx(final_out)
                final_out_dy = self.dy(final_out)
                final_out_dxx = self.dxx(final_out)
                final_out_dyy = self.dyy(final_out)
                # append along the time axis
                out_list.append(final_out)
                dx_list.append(final_out_dx)
                dy_list.append(final_out_dy)
                dxx_list.append(final_out_dxx)
                dyy_list.append(final_out_dyy)

            out_seq = torch.stack(out_list, dim=0)  # [time, batch, 1, 64, 64]
            out_seq = out_seq.permute(1, 0, 2, 3, 4)  # [batch, time, 1, 64, 64]

            dx_seq = torch.stack(dx_list, dim=0)
            dx_seq = dx_seq.permute(1, 0, 2, 3, 4)

            dy_seq = torch.stack(dy_list, dim=0)
            dy_seq = dy_seq.permute(1, 0, 2, 3, 4)

            dxx_seq = torch.stack(dxx_list, dim=0)
            dxx_seq = dxx_seq.permute(1, 0, 2, 3, 4)

            dyy_seq = torch.stack(dyy_list, dim=0)
            dyy_seq = dyy_seq.permute(1, 0, 2, 3, 4)

            return out_seq, dx_seq, dy_seq, dxx_seq, dyy_seq, hidden_list
        else:
            for step in range(self.num_steps):
                # first we select features as mentioned above in the NOTE
                x_input = x[:, step, [0, 1], :, :]
                well_loc, well_para = x[:, step, 3, :, :], x[:, step, 4, :, :]
                well = well_loc * well_para
                well = well.unsqueeze(1)
                x_input = torch.cat((x_input, well), dim=1)

                encoded_x, x32, x64 = self.inception_encode(x_input)
                curr_h, curr_c = self.conv_lstm(encoded_x, hidden_list[step][0], hidden_list[step][1])
                hidden_list.append((curr_h, curr_c))

                final_out = self.decoder(curr_h, x32, x64)  # [batch, 1, 64, 64]

                # append along the time axis
                out_list.append(final_out)

            out_seq = torch.stack(out_list, dim=0)  # [time, batch, 1, 64, 64]
            out_seq = out_seq.permute(1, 0, 2, 3, 4)  # [batch, time, 1, 64, 64]

            return out_seq, hidden_list


# check tensor dimensions
if __name__ == '__main__':
    xxx = torch.randn(16, 24, 6, 64, 64).cuda()
    print(f'xxx shape: {xxx.shape}')
    model = InceptionLSTM(3, 1, 128, 24).cuda()
    model.eval()

    y, ss = model(xxx)
    print(f'y shape: {y.shape}')
    print(ss[0][0].shape)
    print(f'total params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
