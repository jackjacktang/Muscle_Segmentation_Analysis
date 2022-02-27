from ptsemseg.models.utils import *
import functools
import torch.nn.functional as F
from ptsemseg.loss import cross_entropy2d
import torchvision.models as models


class ResAttnUnet(nn.Module):
    def __init__(self,
                 n_classes=2,
                 n_channels=3,
                 is_deconv=True,
                 decoder_kernel_size=3,
                 ):
        super(ResAttnUnet, self).__init__()

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

        # self.decoder4 = DecoderBlock(in_channels=filters[3] + filters[2],
        #                              n_filters=filters[2],
        #                              kernel_size=decoder_kernel_size,
        #                              is_deconv=is_deconv)
        # self.decoder3 = DecoderBlock(in_channels=filters[2] + filters[1],
        #                              n_filters=filters[1],
        #                              kernel_size=decoder_kernel_size,
        #                              is_deconv=is_deconv)
        # self.decoder2 = DecoderBlock(in_channels=filters[1] + filters[0],
        #                              n_filters=filters[0],
        #                              kernel_size=decoder_kernel_size,
        #                              is_deconv=is_deconv)
        # self.decoder1 = DecoderBlock(in_channels=filters[0] + filters[0],
        #                              n_filters=filters[0],
        #                              kernel_size=decoder_kernel_size,
        #                              is_deconv=is_deconv)

        # print(filters)
        self.Up4 = DecoderBlock2(in_channels=filters[3], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = unetConv2(filters[3], filters[2], is_batchnorm=True, padding=1)

        self.Up3 = DecoderBlock2(in_channels=filters[2], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = unetConv2(filters[2], filters[1], is_batchnorm=True, padding=1)

        self.Up2 = DecoderBlock2(in_channels=filters[1], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = unetConv2(filters[1], filters[0], is_batchnorm=True, padding=1)

        self.Up1 = DecoderBlock2(in_channels=filters[0], kernel_size=decoder_kernel_size, is_deconv=is_deconv)
        self.Att1 = Attention_block(F_g=32, F_l=64, F_int=16)
        # self.Up_conv1 = unetConv2(filters[0] + 32, filters[0], is_batchnorm=True, padding=1)

        self.final_up = DecoderBlock(filters[0] + 32, 32, is_deconv=True)
        self.finalconv = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1, bias=False),
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
        # print('x', x.size(), 'x_', x_.size())
        # print('e1', e1.size(), 'e2', e2.size(), 'e3', e3.size(), 'e4', e4.size())

        # center = self.center(e4)
        # d4 = self.decoder4(torch.cat([center, e3], 1))
        # d3 = self.decoder3(torch.cat([d4, e2], 1))
        # d2 = self.decoder2(torch.cat([d3, e1], 1))
        # d1 = self.decoder1(torch.cat([d2, x], 1))

        # print('center: ', center.size())

        # d4 = self.conv4_1(center)
        # d4 = self.norm4_1(d4)
        # d4 = self.relu4_1(d4)
        # d4 = self.deconv4(d4)
        # d4 = self.norm4_2(d4)
        # d4 = self.relu4_2(d4)
        d4 = self.Up4(e4)
        # print('Up4', d4.size())
        e3 = self.Att4(g=d4, x=e3)
        # print('Att4', e4.size())
        d4 = torch.cat((e3, d4), dim=1)
        # print('cat4: ', d4.size())
        d4 = self.Up_conv4(d4)
        # print('d4: ', d4.size())

        d3 = self.Up3(d4)
        # print('Up3', d3.size())
        e2 = self.Att3(g=d3, x=e2)
        # print('Att3', e3.size())
        d3 = torch.cat((e2, d3), dim=1)
        # print('cat3: ', d3.size())
        d3 = self.Up_conv3(d3)
        # print('d4: ', d3.size())

        d2 = self.Up2(d3)
        # print('Up2', d2.size())
        e1 = self.Att2(g=d2, x=e1)
        # print('Att2', e2.size())
        d2 = torch.cat((e1, d2), dim=1)
        # print('cat2: ', d2.size())
        d2 = self.Up_conv2(d2)
        # print('d2: ', d2.size())

        d1 = self.Up1(d2)
        # print('Up1', d1.size(), 'x', x.size())
        x = self.Att1(g=d1, x=x)
        # print('Att1', x.size())
        d1 = torch.cat((x, d1), dim=1)
        # print('cat1: ', d1.size())
        # d1 = self.Up_conv1(d1)
        # print('d1: ', d1.size())
        d1 = self.final_up(d1)
        # print('d1', d1.size())


        f = self.finalconv(d1)
        # print('f', f.size())
        return f
