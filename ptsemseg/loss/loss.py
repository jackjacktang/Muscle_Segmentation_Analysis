DEBUG=False
def log(s):
    if DEBUG:
        print(s)
###################
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

def regression_l1(input, target, weight=None, size_average=True):
    # loss = nn.L1Loss(input, target, size_average=size_average)
    loss = nn.L1Loss(input, target)
    return loss


def cross_entropy2d(input, target, weight=None, size_average=True):
    # print('input: ', input.size())
    # print('target: ', target.size())
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h > ht and w > wt:  # upsample labels
        target = target.unsequeeze(1)
        target = F.upsample(target, size=(h, w), mode="nearest")
        target = target.sequeeze(1)
    elif h < ht and w < wt:  # upsample images
        input = F.upsample(input, size=(ht, wt), mode="bilinear")
    elif h != ht and w != wt:
        raise Exception("Only support upsampling")

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1).long()
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )

    # print(type(loss))
    return loss

def binary_cross_entropy(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h > ht and w > wt:  # upsample labels
        target = target.unsequeeze(1)
        target = F.upsample(target, size=(h, w), mode="nearest")
        target = target.sequeeze(1)
    elif h < ht and w < wt:  # upsample images
        input = F.upsample(input, size=(ht, wt), mode="bilinear")
    elif h != ht and w != wt:
        raise Exception("Only support upsampling")

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1).long()

    target_flip = torch.bitwise_xor(target, 1)
    target = torch.cat((target.unsqueeze_(-1), target_flip.unsqueeze_(-1)), dim=1)
    print(input.size())
    print(target.size())

    loss = nn.BCEWithLogitsLoss()
    return loss(input, target)

class dice_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        smooth = 1.


        #iflat = pred.max(1)[1].view(-1).float()
        # print(pred.size(), target.size())
        iflat = F.softmax(pred, dim=1)[:,1,:].contiguous().view(-1)
        tflat = target.view(-1).float()
        # print(iflat.size(), tflat.size())
        intersection = (iflat * tflat).sum().float()
        dice_score = 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

        return dice_score

def cross_entropy3d(input, target, weight=None, size_average=True):
    log('LOSS=>CrossEntropy3D=>input.size():{} target.size():{}'.format(input.size(), target.size()))
    loss = nn.CrossEntropyLoss(weight=weight, size_average=size_average)
    return loss(input, target)

def multi_scale_cross_entropy2d(
    input, target, weight=None, size_average=True, scale_weight=None
):
    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight == None:  # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp))

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(
            input=inp, target=target, weight=weight, size_average=size_average
        )

    return loss


def bootstrapped_cross_entropy2d(input,
                                  target,
                                  K,
                                  weight=None,
                                  size_average=True):

    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input,
                                   target,
                                   K,
                                   weight=None,
                                   size_average=True):

        n, c, h, w = input.size()
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(input,
                               target,
                               weight=weight,
                               reduce=False,
                               size_average=False,
                               ignore_index=250)

        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(
            input=torch.unsqueeze(input[i], 0),
            target=torch.unsqueeze(target[i], 0),
            K=K,
            weight=weight,
            size_average=size_average,
        )
    return loss / float(batch_size)

class TripletLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    def soft_dice(self, preds, labels):
        smooth = 1.
        iflat = F.softmax(preds, dim=1)[:,1,:].contiguous().view(-1)
        tflat = labels.view(-1).float()
        # print(iflat.size(), tflat.size())
        intersection = (iflat * tflat).sum().float()
        dice_score = 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
        return dice_score

    def forward(self, preds_anchor, preds_pos, preds_neg):
        # dice
        # dice_score = self.soft_dice(preds, labels)

        # triplet
        triplet_score = self.triplet_loss(preds_anchor, preds_pos, preds_neg)

        # return dice_score + 0.5*triplet_score
        return triplet_score
