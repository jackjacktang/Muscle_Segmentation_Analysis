DEBUG = True
def log(s):
    if DEBUG:
        print(s)

import logging
from ptsemseg.augmentations.augmentations import *

logger = logging.getLogger('ptsemseg')

key2aug = {'gamma': AdjustGamma,
           'hue': AdjustHue,
           'brightness': AdjustBrightness,
           'saturation': AdjustSaturation,
           'contrast': AdjustContrast,
           'rcrop': RandomCrop,
           'hflip': RandomHorizontallyFlip,
           'vflip': RandomVerticallyFlip,
           'scale': Scale,
           'rsize': RandomSized,
           'rsizecrop': RandomSizedCrop,
           'rotate': RandomRotateCV2,
           'translate': RandomTranslate,
           'ccrop': CenterCrop,
           'hflip3d': RandomHorizontallyFlip3d,
           'vflip3d': RandomVerticallyFlip3d,
           'iflip3d': RandomFlipInsideOut3d,
           'rotate3d': RandomRotate3d}

def get_composed_augmentations(aug_dict):
    if aug_dict is None:
        logger.info("Using No Augmentations")
        log("Using No Augmentations")
        return None

    augmentations = []
    for aug_key, aug_param in aug_dict.items():
        augmentations.append(key2aug[aug_key](aug_param))
        logger.info("Using {} aug with params {}".format(aug_key, aug_param))
        log("Using {} aug with params {}".format(aug_key, aug_param))
    return Compose(augmentations)


