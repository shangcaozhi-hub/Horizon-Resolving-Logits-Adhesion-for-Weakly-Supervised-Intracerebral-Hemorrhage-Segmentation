import os
from torchvision import transforms
import random
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image
from PIL import Image
from scipy import ndimage
import math
import torch.nn.functional as F

from PIL import ImageFilter, ImageOps, Image

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F


def expand_tensor(tensor,n):
    """
    Binary mask dilation
    tensor: [B,C,H,W]
    """
    kernel=torch.ones(1, 1, 2*n+1, 2*n+1,device=tensor.device)
    out=F.conv2d(tensor,kernel,padding=n)
    return (out>0).float()


class RSNAClsDataset(Dataset):
    def __init__(self, img_name_list_path, rsna_root, train=True):
        csv_path = os.path.join(
            img_name_list_path,
            "train.csv" if train else "val.csv"
        )

        self.data = np.array(pd.read_csv(csv_path))
        self.root = rsna_root
        self.train = train

        # ============================================================
        # 1. Spatial transforms
        #    只负责改变空间位置 / 尺寸
        # ============================================================
        self.train_spatial_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                size=(240, 240),
                scale=(0.8, 1.1),
                ratio=(3 / 4, 4 / 3),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),

            transforms.RandomRotation(
                90,
                interpolation=transforms.InterpolationMode.BILINEAR
            ),

            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])

        self.val_spatial_transform = transforms.Compose([
            transforms.Resize(
                (240, 240),
                interpolation=transforms.InterpolationMode.BILINEAR
            )
        ])

        # ============================================================
        # 2. Intensity transforms
        #    不改变任何空间位置，只改变灰度/强度
        # ============================================================
        self.train_intensity_transform = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3
            ),
            transforms.ToTensor()
        ])

        self.val_intensity_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # mask 只做 tensor 转换，不做任何额外 augmentation
        self.mask_transform = transforms.ToTensor()

    def __getitem__(self, idx):

        name = self.data[idx][0]

        img = Image.open(
            os.path.join(self.root, name + ".png")
        ).convert("L")

        label = torch.tensor(
            self.data[idx][1:2].astype("float32")
        )

        # ============================================================
        # Step 1:
        # 所有空间 augmentation 只执行一次
        # ============================================================
        if self.train:
            spatial_img = self.train_spatial_transform(img)
        else:
            spatial_img = self.val_spatial_transform(img)

        # ============================================================
        # Step 2:
        # 在 intensity augmentation 之前生成 mask
        #
        # 此时 mask 与之后的 img_tensor 空间位置严格一致，
        # 但 mask 不受到 brightness / contrast 影响
        # ============================================================
        mask = self.mask_transform(spatial_img) > 0.05

        # ============================================================
        # Step 3:
        # 仅对网络输入执行 intensity augmentation
        # ============================================================
        if self.train:
            img_tensor = self.train_intensity_transform(spatial_img)
        else:
            img_tensor = self.val_intensity_transform(spatial_img)

        # grayscale -> pseudo RGB
        img_tensor = img_tensor.repeat(3, 1, 1)

        return img_tensor, label, mask

    def __len__(self):
        return len(self.data)


def build_dataset(is_train,args):

    dataset=RSNAClsDataset(
        img_name_list_path=args.img_list,
        rsna_root=args.data_path,
        train=is_train
    )

    return dataset,1

