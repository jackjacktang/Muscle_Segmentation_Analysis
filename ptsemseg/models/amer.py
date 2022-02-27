from ptsemseg.models.utils import *
import functools
import torch.nn.functional as F


class Amer(nn.Module):
    def __init__(self, n_classes=2, n_channels=1):
        super(Amer, self).__init__()
        self.is_bn = True

        self.Maxpool = nn.MaxPool2d(kernel_size=2)

        self.Conv1 = nn.Sequential(
                nn.Conv2d(n_channels, 32, 3, 1, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            )
        self.Conv2 = nn.Sequential(
                nn.Conv2d(32, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )
        self.Dense_latent = nn.Linear(2*2*64, 3)
        self.Dense_latent_up = nn.Linear(3, 2*2*64)
        self.ReLU = nn.ReLU()

        self.Up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.Up1 = nn.ConvTranspose2d(32, n_channels, kernel_size=2, stride=2)


    def forward(self, A, P, N):
        A_p, A_embed = self.forward_helper(A)
        P_p, P_embed = self.forward_helper(P)
        N_p, N_embed = self.forward_helper(N)
        return (A_p, P_p, N_p), (A_embed, P_embed, N_embed)

    def forward_helper(self, x):
        # encoding path
        # print('x', x.size())
        x1 = self.Conv1(x)
        # print('x1', x1.size())

        x2 = self.Maxpool(x1)
        # print('x2', x2.size())
        x2 = self.Conv2(x2)
        # print('x2', x2.size())
        x3 = self.Maxpool(x2)
        # print('x3', x3.size())
        dense1 = x3.view(x3.size(0), -1)
        # print('dense1', dense1.size())

        # latent vector
        dense_latent = self.Dense_latent(dense1)
        # print('dense_latent', dense_latent.size())
        
        # decoding path
        dense_latent_up = self.Dense_latent_up(dense_latent)
        # print('dense_latent_up', dense_latent_up.size())

        d3 = dense_latent_up.view(dense_latent_up.size(0), 64, 2, 2)
        # print('d3', d3.size())
        d2 = self.Up2(d3)
        # print('d2', d2.size())
        d1 = self.Up1(d2)
        # print('d1', d1.size())
        return d1, dense_latent