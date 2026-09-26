import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
import numpy as np
from monai import transforms
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


def flair_to_ncct(flair):
    """Apply a fixed, label-independent NCCT-like transform to FLAIR."""
    flair = flair.clamp(0.0, 1.0)
    brain = flair > 0
    mapped = (0.08 + 0.62 * flair.pow(1.2)).clamp(0.0, 0.75)
    # mapped = (0.065 + 0.89 * flair.pow(2.0)).clamp(0, 0.82)
    # mapped = torch.where(flair > 0, mapped, torch.zeros_like(mapped))
    # mapped = (0.08 + 0.62 * flair.pow(1.2)).clamp(0.0, 0.75)  mapped = (0.08 + 0.73 * flair.pow(1.6)).clamp(0.0, 0.82)
    return torch.where(brain, mapped, torch.zeros_like(mapped))

def expand_tensor(tensor, n):
    """
    扩张张量中的1像素,向外扩张n个像素

    参数:
    tensor (torch.Tensor): 输入的0,1二值张量
    n (int): 扩张的像素数

    返回:
    torch.Tensor: 扩张后的张量
    """
    # 获取输入张量的形状
    batch_size, channels, height, width = tensor.shape

    # 在每个维度上进行扩张
    kernel = torch.ones(1, 1, 2 * n + 1, 2 * n + 1).to(tensor.device)
    padded = F.conv2d(tensor, kernel, padding=n)

    # 将扩张后的结果二值化
    output = (padded > 0).float()

    return output


class TrainDataset(Dataset):
    def __init__(self, data, img_size):
        self.data = data

        ## origin augmentation
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor()])
        # self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        img_path = self.data[index]
        img = Image.open(img_path)
        img = img.convert(mode='RGB')
        pos_1 = self.transform(img)
        pos_2 = self.transform(img)
        if img_path.split(os.path.sep)[-2] == 'tumor':
            label = torch.ones(1)
        else:
            label = torch.zeros(1)
        return pos_1, pos_2, label

    def __len__(self):
        return len(self.data)


class ImageDataset(Dataset):
    def __init__(self, data):
        self.data = data

        ## origin augmentation
        self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        img_path = self.data[index]
        img = Image.open(img_path)
        img = img.convert(mode='RGB')
        if img_path.split(os.path.sep)[-2] == 'tumor':
            label = torch.ones(1)
        else:
            label = torch.zeros(1)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


class Brats2021Dataset(Dataset):
    def __init__(self, file_path, data, img_size, stage, task='cls', ncct=False, fused=False):
        self.data = data
        self.file_path = file_path
        self.task = task
        self.ncct = bool(ncct)
        self.fused = bool(fused)

        ## origin augmentation
        if stage == 'trains':
            # self.transform = transforms.Compose([transforms.Resize(img_size)])
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.RandFlip(prob=0.5, spatial_axis=0),
                transforms.RandFlip(prob=0.5, spatial_axis=1),
                transforms.RandScaleIntensity(factors=0.1, prob=0.5),
                transforms.RandShiftIntensity(offsets=0.1, prob=0.5),
                transforms.RandAdjustContrast(gamma=(0.5, 1.5), prob=0.5)
                ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(img_size)]
            )

        img_size2 = [img_size[0] // 2, img_size[1] // 2]
        self.transform2 = transforms.Compose([
            transforms.Resize(img_size2)])

    def __getitem__(self, index):
        patient_id = self.data[index].split('_')[0] + '_' + self.data[index].split('_')[1]
        img_path = self.file_path + '/vol' + '/' + patient_id + '/' + self.data[index] + '_vol.npy'
        label_path = self.file_path + '/seg' + '/' + patient_id + '/' + self.data[index] + '_seg.npy'

        img = torch.from_numpy(np.load(img_path)).squeeze().float()
        mask = np.load(label_path)
        mask = np.where(mask > 0.0, 1.0, 0.0)
        if mask.mean() > 0.01:
            label = torch.ones(1)
        else:
            label = torch.zeros(1)

        # Stored channel order: FLAIR, T1ce, T1, T2.  Augment FLAIR alone,
        # map its intensity to the NCCT-like range, then repeat it as 3 channels.
        if self.fused:
            skull_mask = (img.mean(1).unsqueeze(0) > 0).float().unsqueeze(0)
            view = self.transform(img)[torch.arange(4) != 2]
        else:
            flair = img[0:1]
            skull_mask = (flair > 0).float().unsqueeze(0)
            source = flair_to_ncct(flair) if self.ncct else flair
            view = self.transform(source).repeat(3, 1, 1)

        info = self.data[index].split('_')[1] + '_slice_' + self.data[index].split('_')[-1]

        if self.task == 'cls':
            return view,  label,  skull_mask
        if self.task == 'seg':
            return {"image": view, "mask": mask.squeeze(-1), "label": label, "case": info}

    def __len__(self):
        return len(self.data)


def show_four_modalities(volume_path, save_path=None, show=True):
    """Display the four channels of one BraTS 2-D ``*_vol.npy`` sample."""
    volume = np.load(volume_path)
    volume = np.squeeze(volume)
    if volume.ndim != 3 or volume.shape[0] != 4:
        raise ValueError(
            f'Expected a four-channel array shaped [4, H, W], got {volume.shape}')

    # The stored BraTS channel order used by this project. Brats2021Dataset
    # selects index 0 (FLAIR), maps its intensity, and repeats it 3 times.
    modality_names = ('FLAIR', 'T1ce', 'T1', 'T2')
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    for channel, (axis, name) in enumerate(zip(axes, modality_names)):
        image = volume[channel]
        nonzero = image[image != 0]
        if nonzero.size:
            vmin, vmax = np.percentile(nonzero, (1, 99))
        else:
            vmin, vmax = 0.0, 1.0
        axis.imshow(image, cmap='gray', vmin=vmin, vmax=max(vmax, vmin + 1e-6))
        axis.set_title(f'Channel {channel}: {name}\n'
                       f'range [{image.min():.3f}, {image.max():.3f}]')
        axis.axis('off')

    fig.suptitle(os.path.basename(volume_path))
    if save_path:
        output_dir = os.path.dirname(os.path.abspath(save_path))
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches='tight')
        print(f'Saved four-modality preview to: {save_path}')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Preview the four modalities of a BraTS slice.')
    parser.add_argument('volume', help='Path to a *_vol.npy file shaped [4, H, W, 1].')
    parser.add_argument('--save', default='', help='Optional path used to save the preview image.')
    parser.add_argument('--no-show', action='store_true', help='Save without opening a plot window.')
    arguments = parser.parse_args()
    show_four_modalities(arguments.volume, arguments.save or None, not arguments.no_show)

