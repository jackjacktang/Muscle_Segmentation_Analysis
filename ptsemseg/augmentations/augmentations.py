# Adapted from https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py

import math
import numbers
import random
import numpy as np
import torch
import torchvision.transforms.functional as tf
import cv2

from PIL import Image, ImageOps


class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations
        self.PIL2Numpy = False

    def __call__(self, img, mask):
        assert img.shape[:2] == mask.shape[:2]
        for a in self.augmentations:
            img, mask = a(img, mask)

        return img, mask
class RandomHorizontallyFlip3d(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask):
        # if random.random() < self.p:
        #     return (
        #         np.fliplr(img), np.fliplr(mask)
        #     )
        # return img, mask
        if random.random() < self.p:
            return (
                torch.fliplr(img), torch.fliplr(mask)
            )
        return img, mask


class RandomVerticallyFlip3d(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask):
        # def __call__(self, img, mask):
    #     if random.random() < self.p:
    #         return (
    #             (np.flipud(img)).copy(), (np.flipud(mask)).copy()
    #         )
    #     return img, mask
        if random.random() < self.p:
            return (
                (torch.flipud(img)).copy(), (torch.flipud(mask)).copy()
            )
        return img, mask

class RandomFlipInsideOut3d(object):
    def __init__(self, p):
        self.p = p # [0,1)

    def __call__(self, img, mask):
        # if random.random() < self.p:
        #     return (
        #         (np.flip(img, axis=2)).copy(), (np.flip(mask, axis=2)).copy()
        #     )
        # return img, mask
        if random.random() < self.p:
            return (
                (np.flip(img, axis=2)).copy(), (np.flip(mask, axis=2)).copy()
            )
        return img, mask


class RandomRotate3d(object):
    def __init__(self, degree):
        self.degree = degree # -180, 180

    def __call__(self, img, mask):
        # from scipy.ndimage import rotate
        # rotate_degree = random.random() * 2 * self.degree - self.degree # -degree, +degree
        # rotation_plane = random.sample(range(0, 3), 2)
        # return (
        #     rotate(img, rotate_degree, rotation_plane).copy(),
        #     rotate(mask, rotate_degree, rotation_plane).copy()
        #     )
        from scipy.ndimage import rotate
        rotate_degree = random.random() * 2 * self.degree - self.degree # -degree, +degree
        return (
            torch.from_numpy(rotate(img, rotate_degree, reshape=False)),
            torch.from_numpy(rotate(mask, rotate_degree, reshape=False))
            )

class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return (
                img.resize((tw, th), Image.BILINEAR),
                mask.resize((tw, th), Image.NEAREST),
            )

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return (
            img.crop((x1, y1, x1 + tw, y1 + th)),
            mask.crop((x1, y1, x1 + tw, y1 + th)),
        )
class AdjustGamma(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, img, mask):
        assert img.size() == mask.size()
        return tf.adjust_gamma(img, random.uniform(1, 1 + self.gamma)), mask


class AdjustSaturation(object):
    def __init__(self, saturation):
        self.saturation = saturation

    def __call__(self, img, mask):
        assert img.size() == mask.size()
        return tf.adjust_saturation(img,
                                    random.uniform(1 - self.saturation,
                                                   1 + self.saturation)), mask


class AdjustHue(object):
    def __init__(self, hue):
        self.hue = hue

    def __call__(self, img, mask):
        assert img.size() == mask.size()
        return tf.adjust_hue(img, random.uniform(-self.hue,
                                                 self.hue)), mask


class AdjustBrightness(object):
    def __init__(self, bf):
        self.bf = bf

    def __call__(self, img, mask):
        assert img.size() == mask.size()
        return tf.adjust_brightness(img,
                                    random.uniform(1 - self.bf,
                                                   1 + self.bf)), mask


class AdjustContrast(object):
    def __init__(self, cf):
        self.cf = cf

    def __call__(self, img, mask):
        assert img.size() == mask.size()
        return tf.adjust_contrast(img,
                                  random.uniform(1 - self.cf,
                                                 1 + self.cf)), mask


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return (
            img.crop((x1, y1, x1 + tw, y1 + th)),
            mask.crop((x1, y1, x1 + tw, y1 + th)),
        )


class RandomHorizontallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            # img = cv2.flip(img, 0)
            # mask = cv2.flip(mask, 0)
            img = torch.fliplr(img)
            mask = torch.fliplr(mask)
            return (
                # img.transpose(Image.FLIP_LEFT_RIGHT),
                # mask.transpose(Image.FLIP_LEFT_RIGHT),
                img,
                mask,
            )
        # print('hflip: ', img.shape)
        # print('hflip: ', mask.shape)
        return img, mask


class RandomVerticallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            # img = cv2.flip(img, 1)
            # mask = cv2.flip(mask, 1)
            img = torch.flipud(img)
            mask = torch.flipud(mask)
            return (
                # img.transpose(Image.FLIP_TOP_BOTTOM),
                # mask.transpose(Image.FLIP_TOP_BOTTOM),
                img,
                mask,
            )
        return img, mask


class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask):
        assert img.size == mask.size
        return (
            img.resize(self.size, Image.BILINEAR),
            mask.resize(self.size, Image.NEAREST),
        )


class RandomTranslate(object):
    def __init__(self, offset):
        self.offset = offset  # tuple (delta_x, delta_y)

    def __call__(self, img, mask):
        assert img.size == mask.size
        x_offset = int(2 * (random.random() - 0.5) * self.offset[0])
        y_offset = int(2 * (random.random() - 0.5) * self.offset[1])

        x_crop_offset = x_offset
        y_crop_offset = y_offset
        if x_offset < 0:
            x_crop_offset = 0
        if y_offset < 0:
            y_crop_offset = 0

        cropped_img = tf.crop(img,
                              y_crop_offset,
                              x_crop_offset,
                              img.size[1] - abs(y_offset),
                              img.size[0] - abs(x_offset))

        if x_offset >= 0 and y_offset >= 0:
            padding_tuple = (0, 0, x_offset, y_offset)

        elif x_offset >= 0 and y_offset < 0:
            padding_tuple = (0, abs(y_offset), x_offset, 0)

        elif x_offset < 0 and y_offset >= 0:
            padding_tuple = (abs(x_offset), 0, 0, y_offset)

        elif x_offset < 0 and y_offset < 0:
            padding_tuple = (abs(x_offset), abs(y_offset), 0, 0)

        return (
            tf.pad(cropped_img,
                   padding_tuple,
                   padding_mode='reflect'),
            tf.affine(mask,
                      translate=(-x_offset, -y_offset),
                      scale=1.0,
                      angle=0.0,
                      shear=0.0,
                      fillcolor=250))


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        # rotate_degree = random.random() * 2 * self.degree - self.degree
        # return (
        #     tf.affine(img,
        #               translate=(0, 0),
        #               scale=1.0,
        #               angle=rotate_degree,
        #               resample=Image.BILINEAR,
        #               fillcolor=(0, 0, 0),
        #               shear=0.0),
        #     tf.affine(mask,
        #               translate=(0, 0),
        #               scale=1.0,
        #               angle=rotate_degree,
        #               resample=Image.NEAREST,
        #               fillcolor=250,
        #               shear=0.0))
        from scipy.ndimage import rotate
        rotate_degree = random.random() * 2 * self.degree - self.degree # -degree, +degree
        return (
            torch.from_numpy(rotate(img, rotate_degree, reshape=False)),
            torch.from_numpy(rotate(mask, rotate_degree, reshape=False))
            )


class RandomRotateCV2(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        """Rotate the image.

        Rotate the image such that the rotated image is enclosed inside the tightest
        rectangle. The area not occupied by the pixels of the original image is colored
        black.

        Parameters
        ----------

        image : numpy.ndarray
            numpy image

        angle : float
            angle by which the image is to be rotated

        Returns
        -------

        numpy.ndarray
            Rotated Image

        """
        # grab the dimensions of the image and then determine the
        # centre
        rotate_degree = random.random() * 2 * self.degree - self.degree
        (h, w) = img.shape[:2]
        # print(img.shape)
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), rotate_degree, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        img = cv2.warpAffine(img, M, (nW, nH))
        lbl = cv2.warpAffine(mask, M, (nW, nH))

        img = cv2.resize(img, (h,w))
        lbl = cv2.resize(lbl, (h,w))

        # print(img.shape)

        #    image = cv2.resize(image, (w,h))
        return img, lbl


class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, mask
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return (
                img.resize((ow, oh), Image.BILINEAR),
                mask.resize((ow, oh), Image.NEAREST),
            )
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return (
                img.resize((ow, oh), Image.BILINEAR),
                mask.resize((ow, oh), Image.NEAREST),
            )


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert img.size == (w, h)

                return (
                    img.resize((self.size, self.size), Image.BILINEAR),
                    mask.resize((self.size, self.size), Image.NEAREST),
                )

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, mask))


class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, img, mask):
        assert img.size == mask.size

        w = int(random.uniform(0.5, 2) * img.size[0])
        h = int(random.uniform(0.5, 2) * img.size[1])

        img, mask = (
            img.resize((w, h), Image.BILINEAR),
            mask.resize((w, h), Image.NEAREST),
        )

        return self.crop(*self.scale(img, mask))
