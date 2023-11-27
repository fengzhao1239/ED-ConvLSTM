import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, in_channels, out_channels, features=[16, 32, 64,]): # [16, 32, 64, 128]
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2) # original : maxpool

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(in_channels=feature*2, out_channels=feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottom = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        # x = x[:, [0, 1, 3, 4, 5], :, :]
        x_input = x[:, [0, 1, 5], :, :]
        well_loc, well_para = x[:, 3, :, :], x[:, 4, :, :]
        well = well_loc * well_para
        well = well.unsqueeze(1)
        x = torch.cat((x_input, well), dim=1)

        skip_connections = []

        # down sampling
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottom(x)
        skip_connections = skip_connections[::-1]    # reverse list

        # up sampling
        # notice: we do up + DoubleConv per step
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            # check if the two cat tensors match during skip connection
            # if x.shape != skip_connection.shape:
            #     x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


class Dx(nn.Module):
    def __init__(self, dx_filter, in_channel, out_channel):
        super(Dx, self).__init__()
        self.conv_dx = nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='replicate', groups=in_channel, bias=False)
        self.conv_dx.weight = nn.Parameter(torch.FloatTensor(dx_filter), requires_grad=False)

    def forward(self, x):
        dx_value = self.conv_dx(x)
        return dx_value


class Dxx(nn.Module):
    def __init__(self, dx_filter, in_channel, out_channel):
        super(Dxx, self).__init__()
        self.conv_dx = nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='replicate', groups=in_channel, bias=False)
        self.conv_dx.weight = nn.Parameter(torch.FloatTensor(dx_filter), requires_grad=False)

    def forward(self, x):
        dx_value = self.conv_dx(x)
        # dxx_value = self.conv_dx(dx_value)
        return dx_value


class Dy(nn.Module):
    def __init__(self, dy_filter, in_channel, out_channel):
        super(Dy, self).__init__()
        self.conv_dy = nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='replicate', groups=in_channel, bias=False)
        self.conv_dy.weight = nn.Parameter(torch.FloatTensor(dy_filter), requires_grad=False)

    def forward(self, x):
        dy_value = self.conv_dy(x)
        return dy_value


class Dyy(nn.Module):
    def __init__(self, dy_filter, in_channel, out_channel):
        super(Dyy, self).__init__()
        self.conv_dy = nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='replicate', groups=in_channel, bias=False)
        self.conv_dy.weight = nn.Parameter(torch.FloatTensor(dy_filter), requires_grad=False)

    def forward(self, x):
        dy_value = self.conv_dy(x)
        # dyy_value = self.conv_dy(dy_value)
        return dy_value

def test():
    x = torch.randn((16, 6, 64, 64))
    model = UNET(in_channels=4, out_channels=1)
    preds = model(x)
    # assert preds.shape == x.shape
    print(x.shape)
    print(preds.shape)
    print(f'total params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')


if __name__ == "__main__":
    test()
