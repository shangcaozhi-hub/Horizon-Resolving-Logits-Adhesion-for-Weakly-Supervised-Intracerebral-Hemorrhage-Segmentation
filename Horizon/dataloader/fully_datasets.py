from torch.utils.data import Dataset
import torch
import h5py
from torchvision.transforms import Resize, CenterCrop , InterpolationMode
import numpy as np
from torchvision import transforms
from PIL import Image

class insDataSet(Dataset):
    def __init__(
            self,
            base_dir=None,
            data_ls=None, train=False):
        self._base_dir = base_dir
        self.ls = data_ls
        self.train = train

        if self.train:
            self.transforms_Jitter = transforms.Compose([
                transforms.ColorJitter(brightness=0.3, contrast=0.3)
            ])

            self.transforms = transforms.Compose([
                transforms.RandomRotation(degrees=90),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
            ])

        with open(self._base_dir + self.ls, "r") as f1:
            self.sample_list = f1.readlines()
            # 将train/test文件转化为列表
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        print("seg dataset  total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        list = self.sample_list
        case = self.sample_list[idx]
        h5f = h5py.File(self._base_dir + case, "r")
        image = torch.from_numpy(h5f["image"][:])
        mask = torch.from_numpy(h5f["label"][:])
        image = torch.unsqueeze(image, 0)
        mask = torch.unsqueeze(mask, 0)

        torch_resize = Resize((240, 240))  # 定义Resize类对象
        mask_resize = Resize((240, 240), interpolation=InterpolationMode.NEAREST_EXACT)

        if self.train:
            image = torch_resize(image)
            orign_mask = mask
            mask = mask_resize(mask)
            data = torch.cat([image, mask], dim=0)
            data = self.transforms(data)
            image = self.transforms_Jitter(data[0, :, :].unsqueeze(0))
            mask = data[1, :, :].unsqueeze(0)
            mask = (mask > 0).float()
        else:
            image = torch_resize(image)
            orign_mask = mask
            mask = mask_resize(mask)
            mask = (mask > 0).float()
        #  "orign_label": orign_label.squeeze(0)

        if mask.mean() > 0.01:
            label = torch.ones(1)
        else:
            label = torch.zeros(1)

        sample = {"image": image.repeat(3, 1, 1), "mask": mask.squeeze(0), "label": label, "case": case}
        return sample


class bhsdDataSet(Dataset):
    def __init__(
            self,
            base_dir=None,
            data_ls=None, train=False):
        self._base_dir = base_dir
        self.ls = data_ls
        self.train = train
        if self.train:

            self.transforms_Jitter = transforms.Compose([
                transforms.ColorJitter(brightness=0.3, contrast=0.3)
            ])

            self.transforms = transforms.Compose([
                transforms.RandomRotation(degrees=90),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
            ])

        with open(self._base_dir + self.ls, "r") as f1:
            self.sample_list = f1.readlines()
            # 将train/test文件转化为列表
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        print("seg dataset  total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        list = self.sample_list
        case = self.sample_list[idx]
        img_path = self._base_dir + '/label_192/vol/' + case + '_vol.npy'
        label_path = self._base_dir + '/label_192/seg/' + case + '_seg.npy'
        image = torch.from_numpy(np.load(img_path)).squeeze().float()
        mask = torch.from_numpy(np.load(label_path)).squeeze().float()
        image = torch.unsqueeze(image, 0)
        mask = torch.unsqueeze(mask, 0)

        torch_resize = Resize((240, 240))  # 定义Resize类对象
        mask_resize = Resize((240, 240), interpolation=0)

        if self.train:
            image = torch_resize(image)
            orign_mask = mask
            mask = mask_resize(mask)
            data = torch.cat([image, mask], dim=0)
            data = self.transforms(data)
            image = self.transforms_Jitter(data[0, :, :].unsqueeze(0))
            mask = data[1, :, :].unsqueeze(0)
            mask = (mask > 0).float()
        else:
            image = torch_resize(image)
            orign_mask = mask
            mask = mask_resize(mask)
            mask = (mask > 0).float()

        if mask.mean() > 0.01:
            label = torch.ones(1)
        else:
            label = torch.zeros(1)

            #  "orign_label": orign_label.squeeze(0)
        sample = {"image": image.repeat(3, 1, 1), "mask": mask.squeeze(0), "label": label, "case": case}

        return sample

