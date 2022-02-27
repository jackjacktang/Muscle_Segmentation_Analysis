from ptsemseg.models.utils import *
import functools
import torch.nn.functional as F
from ptsemseg.loss import cross_entropy2d
import torchvision.models as models


class ResNet34Unet(nn.Module):
    def __init__(self,
                 n_classes=2,
                 n_channels=3,
                 is_deconv=True,
                 decoder_kernel_size=3,
                 ):
        super(ResNet34Unet, self).__init__()

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.base_size = 512
        self.crop_size = 512
        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}
        # self.firstconv = resnet.conv1
        # assert num_channels == 3, "num channels not used now. to use changle first conv layer to support num channels other then 3"
        # try to use 8-channels as first input
        if n_channels == 3:
            self.firstconv = resnet.conv1
        else:
            self.firstconv = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.center = DecoderBlock(in_channels=filters[3],
                                   n_filters=filters[3],
                                   kernel_size=decoder_kernel_size,
                                   is_deconv=is_deconv)
        self.decoder4 = DecoderBlock(in_channels=filters[3] + filters[2],
                                     n_filters=filters[2],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder3 = DecoderBlock(in_channels=filters[2] + filters[1],
                                     n_filters=filters[1],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder2 = DecoderBlock(in_channels=filters[1] + filters[0],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder1 = DecoderBlock(in_channels=filters[0] + filters[0],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)

        self.finalconv = nn.Sequential(nn.Conv2d(filters[0], 32, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(),
                                       nn.Dropout2d(0.1, False),
                                       nn.Conv2d(32, n_classes, 1))

    def require_encoder_grad(self, requires_grad):
        blocks = [self.firstconv,
                  self.encoder1,
                  self.encoder2,
                  self.encoder3,
                  self.encoder4]

        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

    def forward(self, x):
        # stem
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x_ = self.firstmaxpool(x)

        # Encoder
        e1 = self.encoder1(x_)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        center = self.center(e4)

        d4 = self.decoder4(torch.cat([center, e3], 1))
        d3 = self.decoder3(torch.cat([d4, e2], 1))
        d2 = self.decoder2(torch.cat([d3, e1], 1))
        d1 = self.decoder1(torch.cat([d2, x], 1))
        # print('e1', e1.size(), 'e2', e2.size(), 'e3', e3.size(), 'e4', e4.size())
        # print('center', center.size())
        # print('d1', d1.size(), 'd2', d2.size(), 'd3', d3.size(), 'd4', d4.size())

        f = self.finalconv(d1)
        return f
