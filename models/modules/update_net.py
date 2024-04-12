import torch
import torch.nn as nn


class UpdateNet(nn.Module):
    def __init__(self, in_c=256, out_c=256):
        super(UpdateNet, self).__init__()
        self.update = nn.Sequential(
            nn.Conv2d(in_c*3, out_c, (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, (1, 1))
        )

    def forward(self, input, init, cur):
        x = torch.cat([input, init, cur], dim=1)
        out = self.update(x)
        out += init
        return out


class UpdateNetV2(nn.Module):
    def __init__(self, in_c=256, out_c=256):
        super(UpdateNetV2, self).__init__()
        self.update = nn.Sequential(
            nn.Conv2d(in_c*3, out_c, (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, (1, 1))
        )

    def forward(self, input, init, cur):
        x = torch.cat([input, init, cur], dim=1)
        out = self.update(x)
        return out


class UpdateNetV3(nn.Module):
    def __init__(self, in_c=256, out_c=256):
        super(UpdateNetV3, self).__init__()
        self.update = nn.Sequential(
            nn.Conv2d(in_c*2, out_c, (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, (1, 1))
        )

    def forward(self, input, init, cur):
        x = torch.cat([input, cur], dim=1)
        out = self.update(x)
        return out

if __name__ == '__main__':
    batch_size = 1
    channels = 256
    x = torch.rand(batch_size, channels, 7, 7).cuda()
    x0 = torch.rand(batch_size, channels, 7, 7).cuda()
    model = UpdateNet().cuda()

    out = model(x, x0, x0)
    print(out.shape)