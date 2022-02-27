import copy
import torchvision.models as models

# from ptsemseg.models.fcn import *
# from ptsemseg.models.segnet import *
# from ptsemseg.models.unet import *
# from ptsemseg.models.unet3d import *
# from ptsemseg.models.xnet import *
# from ptsemseg.models.pspnet import *
# from ptsemseg.models.icnet import *
# from ptsemseg.models.linknet import *
# from ptsemseg.models.linknet3d import *
# from ptsemseg.models.frrn import *
from ptsemseg.models.fcn import *
# from ptsemseg.models.unet3d import *
from ptsemseg.models.unet import *
from ptsemseg.models.pspnet import *
from ptsemseg.models.resunet import  *
from ptsemseg.models.res_attn_unet import*
from ptsemseg.models.amer import*


def get_model(model_dict, n_classes, version=None):
    name = model_dict['arch']
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop('arch')

    if name in ["frrnA", "frrnB"]:
        model = model(n_classes, **param_dict)

    elif name in ["fcn32s", "fcn16s", "fcn8s"]:
        model = model(n_classes=n_classes, **param_dict)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == "segnet":
        model = model(n_classes=n_classes, **param_dict)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == "unet":
        model = model(n_classes=n_classes, **param_dict)
    elif name == "atten_unet":
        model = model(n_classes=n_classes, atten=True, **param_dict)
    elif name == "xnet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "pspnet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "icnet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "icnetBN":
        model = model(n_classes=n_classes, **param_dict)

    else:
        model = model(n_classes=n_classes, **param_dict)

    return model


def _get_model_instance(name):
    try:
        return {
            "fcn32s": fcn32s,
            "fcn8s": fcn8s,
            "fcn16s": fcn16s,
            "atten_unet": unet,
            "resunet": ResNet34Unet,
            "unet": unet,
            "pspnet": PSPNet,
            "resAttUnet": ResAttnUnet,
            "amer": Amer,
            # "unet3d": unet3d,
            # "xnet": xnet,
            # "segnet": segnet,
            # "pspnet": pspnet,
            # "unet3d": unet3d


        }[name]
    except:
        raise("Model {} not available".format(name))
