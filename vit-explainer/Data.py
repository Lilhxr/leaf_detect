from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import WeightedRandomSampler
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from Cutmix import CutMix
import albumentations as A
from typing import Dict
import numpy as np
import random
import torch


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)
np.random.seed(0)
random.seed(0)


class Transforms:
    '''adapt torchvision transform'''

    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))


class ImageDataset(Dataset):
    '''adapt torchvision dataset'''

    def __init__(self, image_folder: ImageFolder):
        self.image_folder = image_folder
        self.classes = self.image_folder.classes
        self.targets = self.image_folder.targets

    def __getitem__(self, index):
        img, lb = self.image_folder[index]
        return (img['image'], lb)

    def __len__(self):
        return len(self.image_folder)

    def __repr__(self):
        return self.image_folder.__repr__()


class DataGenerator(object):
    '''data generator used for generating dataloader'''

    def __init__(self, train_path: str,
                 valid_path: str,
                 test_path: str,
                 batch_size: int,
                 image_size: int,
                 apply_cutmix: bool):
        '''
        :param train_path: (str) path of train data
        :param valid_path: (str) path of valid data
        :param test_path: (str) path of test data
        :param batch_size: (int) batch size
        :param image_size: (int) image dimension
        :param apply_cutmix: (bool) whether apply cutmix
        '''
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.apply_cutmix = apply_cutmix
        self.train_augs = Transforms(A.Compose([
            A.RandomResizedCrop(
                height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD),
            A.CoarseDropout(p=0.5),
            ToTensorV2(),
        ]))
        self.valid_augs = Transforms(A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD),
            ToTensorV2(),
        ]))

    def generate_dataloader(self) -> Dict:

        train_dataloader = None
        valid_dataloader = None
        test_dataloader = None

        train_data = ImageDataset(ImageFolder(
            self.train_path, transform=self.train_augs))

        if self.apply_cutmix:
            train_data = CutMix(train_data, num_class=5, beta=1.0,
                                prob=0.5, num_mix=2)

        class_sample_counts = np.empty(len(train_data.classes), dtype=int)
        for idx in range(len(train_data.classes)):
            class_sample_counts[idx] = train_data.targets.count(idx)
        weights = 1.0 / torch.tensor(class_sample_counts, dtype=torch.float)
        samples_weights = weights[train_data.targets]
        sampler = WeightedRandomSampler(
            weights=samples_weights,
            num_samples=len(samples_weights),
            replacement=True)

        train_dataloader = DataLoader(train_data,
                                      batch_size=self.batch_size,
                                      sampler=sampler,
                                      worker_init_fn=seed_worker,
                                      generator=g)

        valid_data = ImageDataset(ImageFolder(
            self.valid_path, transform=self.valid_augs))
        valid_dataloader = DataLoader(valid_data,
                                      batch_size=self.batch_size,
                                      worker_init_fn=seed_worker,
                                      generator=g)

        test_data = ImageDataset(ImageFolder(
            self.test_path, transform=self.valid_augs))
        test_dataloader = DataLoader(test_data,
                                     batch_size=self.batch_size,
                                     worker_init_fn=seed_worker,
                                     generator=g)

        return {
            'train_dataloader': train_dataloader,
            'valid_dataloader': valid_dataloader,
            'test_dataloader': test_dataloader
        }
