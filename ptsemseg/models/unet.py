from ptsemseg.models.utils import *
import functools
import torch.nn.functional as F
from ptsemseg.loss import cross_entropy2d


class unet(nn.Module):
    def __init__(self, n_classes=2, n_channels=1, atten=False):
        super(unet, self).__init__()
        self.is_bn = True
        self.is_deconv = True
        self.atten = atten

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = unetConv2(n_channels, 64, self.is_bn, padding=1)
        self.Conv2 = unetConv2(64, 128, self.is_bn, padding=1)
        self.Conv3 = unetConv2(128, 256, self.is_bn, padding=1)
        self.Conv4 = unetConv2(256, 512, self.is_bn, padding=1)
        self.Conv5 = unetConv2(512, 1024, self.is_bn, padding=1)

        # self.Up5 = unetUp(1024, 512, self.is_deconv)
        self.Up5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = unetConv2(1024, 512, self.is_bn, padding=1)

        # self.Up4 = unetUp(512, 256, self.is_deconv)
        self.Up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = unetConv2(512, 256, self.is_bn, padding=1)

        # self.Up3 = unetUp(256, 128, self.is_deconv)
        self.Up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = unetConv2(256, 128, self.is_bn, padding=1)

        # self.Up2 = unetUp(128, 64, self.is_deconv)
        self.Up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = unetConv2(128, 64, self.is_bn, padding=1)

        self.Conv_1x1 = nn.Conv2d(64, n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        # print('x1', x1.size())

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        # print('x2', x2.size())

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        # print('x3', x3.size())

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        # print('x4', x4.size())

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        # print('x5', x5.size())

        # decoding + concat path
        d5 = self.Up5(x5)
        # print('d5', d5.size(), 'x4', x4.size())
        if self.atten:
            # print()
            x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        # print('cat d5', d5.size())
        d5 = self.Up_conv5(d5)
        # print('final d5', d5.size())

        d4 = self.Up4(d5)
        # print('d4', d4.size(), 'x3', x3.size())
        if self.atten:
            x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        # print('cat d4', d4.size())
        d4 = self.Up_conv4(d4)
        # print('final d4', d4.size())

        d3 = self.Up3(d4)
        # print('d3', d3.size(), 'x2', x2.size())
        if self.atten:
            x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        # print('cat d3', d3.size())
        d3 = self.Up_conv3(d3)
        # print('final d3', d3.size())

        d2 = self.Up2(d3)
        # print('d2', d2.size(), 'x1', x1.size())
        if self.atten:
            x1 = self.Att2(g=d2, x=x1)
        # print('cat d2', d2.size())
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        # print('final d2', d2.size())

        d1 = self.Conv_1x1(d2)

        return d1