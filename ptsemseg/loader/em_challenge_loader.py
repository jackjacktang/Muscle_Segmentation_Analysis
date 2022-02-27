import os
import torch
import numpy as np
from collections import defaultdict

from torch.utils import data
from PIL import Image

# from ptsemseg.utils import recursive_glob
# from ptsemseg.augmentations import Compose, RandomHorizontallyFlip, RandomRotate, Scale


class emChallengeLoader(data.Dataset):
    def __init__(
        self,
        root,
        split="train",
        img_size=(512, 512),
        is_transform=True,
        augmentations=None,
        test_mode=False,
        n_classes=2
    ):
        """__init__

        :param root:
        :param split:
        :param img_size:
        """
        self.root = root
        self.split = split
        self.img_size = img_size
        self.is_transform = is_transform
        self.n_classes = n_classes # TODO: change to config
        self.augmentations = augmentations
        self.files = defaultdict(list)

        if not test_mode:
            for split in ["train", "test", "val"]:
                file_list = os.listdir(root + "/" + split + "/images")
                self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index][-6:]
        img_path = self.root + "/" + self.split + "/images/" + "train-volume" + img_name
        lbl_path = self.root + "/" + self.split + "/labels/" + "train-labels" + img_name

        img = Image.open(img_path)
        lbl = Image.open(lbl_path)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)
        

        return img, lbl

    def transform(self, img, lbl):
        img = np.array(img)
        lbl = np.array(lbl)

        img = img/255
        lbl = np.where(lbl > 128, 1, 0)

        img = torch.from_numpy(img).float().unsqueeze_(0)
        # lbl = torch.from_numpy(lbl).long()
        # lbl = torch.from_numpy(lbl).float().unsqueeze_(0)
        lbl = torch.from_numpy(lbl).float()

        # print(img.size())
        # print(lbl.size())
        # print(img.unique())
        # print(lbl.unique())
        return img, lbl




if __name__ == "__main__":
    import matplotlib.pyplot as plt

    local_path = "/Users/ultrashieldrog/Datasets/EM_Challenge"
    dst = emChallengeLoader(local_path, is_transform=True)
    bs = 1
    trainloader = data.DataLoader(dst, batch_size=bs)

    # print("Dataset size:", len(dst))
    fig = plt.figure()

    # for i, [img, lbl] in enumerate(trainloader):
    #     print()
        # print(img.shape, lbl.shape)

        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(img[0])
        # ax[1].imshow(lbl[0])

        # plt.show()





