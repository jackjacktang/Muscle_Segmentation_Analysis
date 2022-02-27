import copy
import logging
import functools
from ptsemseg.loss.loss import cross_entropy2d
from ptsemseg.loss.loss import cross_entropy3d
from ptsemseg.loss.loss import bootstrapped_cross_entropy2d
from ptsemseg.loss.loss import multi_scale_cross_entropy2d
from ptsemseg.loss.loss import regression_l1
from ptsemseg.loss.loss import dice_loss
from ptsemseg.loss.loss import binary_cross_entropy
from ptsemseg.loss.loss import TripletLoss


logger = logging.getLogger('ptsemseg')

key2loss = {'cross_entropy': cross_entropy2d,
            'cross_entropy3d': cross_entropy3d,
            'bootstrapped_cross_entropy': bootstrapped_cross_entropy2d,
            'multi_scale_cross_entropy': multi_scale_cross_entropy2d,
            'regression_l1': regression_l1,
            'dice': dice_loss,
            'binary_cross_entropy': binary_cross_entropy,
            'triplet': TripletLoss,
            }


def get_loss_function(cfg):
    if cfg['training']['loss']['name'] == 'cross_entropy2d':
        logger.info("Using default cross entropy loss")
        return key2loss[cfg['training']['loss']['name']]

    elif cfg['training']['loss']['name'] in ['dice', 'binary_cross_entropy', 'triplet']:
        return key2loss[cfg['training']['loss']['name']]()

    else:
        loss_dict = cfg['training']['loss']
        loss_name = loss_dict['name']

        loss_params = {k:v for k,v in loss_dict.items() if k != 'name'}

        if loss_name not in key2loss:
            raise NotImplementedError('Loss {} not implemented'.format(loss_name))

        logger.info('Using {} with {} params'.format(loss_name,
                                                     loss_params))
        return functools.partial(key2loss[loss_name], **loss_params)
