import os.path as pt
import numpy as np
import torch
import random
import torchvision.transforms as transforms
from fcdd.datasets.bases import TorchvisionDataset
from fcdd.datasets.online_superviser import OnlineSuperviser
from fcdd.datasets.preprocessing import local_contrast_normalization_func, local_contrast_normalization_func2, \
    MultiCompose, AWGN, BlackCenter, TargetTransFunctor, remove_red_lines, CLAHE, remove_glare
from fcdd.util.logging import Logger
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_tensor, to_pil_image, normalize
import cv2 as cv

from PIL import Image


class ADKeen(TorchvisionDataset):
    base_folder = 'keen'

    def __init__(self, root: str, normal_class: int, preproc: str, nominal_label: int,
                 supervise_mode: str, noise_mode: str, oe_limit: int, online_supervision: bool, logger: Logger = None):
        """
        AD dataset for KEEN.
        :param root: root directory where data is found or is to be downloaded to
        :param normal_class: the class considered nominal
        :param preproc: the kind of preprocessing pipeline
        :param nominal_label: the label that marks nominal samples in training. The scores in the heatmaps always
            rate label 1, thus usually the nominal label is 0, s.t. the scores are anomaly scores.
        :param supervise_mode: the type of generated artificial anomalies.
            See :meth:`fcdd.datasets.bases.TorchvisionDataset._generate_artificial_anomalies_train_set`.
        :param noise_mode: the type of noise used, see :mod:`fcdd.datasets.noise_mode`.
        :param oe_limit: limits the number of different anomalies in case of Outlier Exposure (defined in noise_mode)
        :param online_supervision: whether to sample anomalies online in each epoch,
            or offline before training (same for all epochs in this case).
        :param logger: logger
        """
        root = pt.join(root, self.base_folder)
        super().__init__(root, logger=logger)


        self.n_classes = 2  # 0: normal, 1: outlier
        self.shape = (1, 224, 224)
        self.raw_shape = (3, 512, 256)
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(2))
        self.outlier_classes.remove(normal_class)
        assert nominal_label in [0, 1]
        self.nominal_label = nominal_label
        self.anomalous_label = 1 if self.nominal_label == 0 else 0

        # mean and std of original pictures per class
        self.mean = np.array([0.51])
        self.std = np.array([0.1482])

        if self.nominal_label != 0:
            print('Swapping labels, i.e. anomalies are 0 and nominals are 1.')

        # different types of preprocessing pipelines, 'lcn' is for using LCN, 'aug{X}' for augmentations
        # also contains options for the black center experiments
        all_transform = []
        if preproc in ['', None, 'default', 'none']:
            test_transform = transform = transforms.Compose([
                transforms.Resize((self.shape[-2], self.shape[-1])),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        elif preproc in ['aug1']:
            test_part_transform = transforms.Compose([
                transforms.Resize(self.raw_shape[-2:]),
            ])

            test_transform = transforms.Compose([
                transforms.Lambda(remove_red_lines),
                transforms.Lambda(remove_glare),
                transforms.Grayscale(),
                transforms.Lambda(CLAHE()),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
            transform = transforms.Compose([
                transforms.Resize(self.raw_shape[-2:]),
                transforms.Lambda(remove_red_lines),
                transforms.Lambda(remove_glare),
                transforms.Grayscale(),
                transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
                transforms.Lambda(CLAHE()),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(self.shape[-1]),
                transforms.ToTensor(),
                transforms.Lambda(AWGN(0.001)),
                transforms.Normalize(self.mean, self.std)
            ])
        elif preproc in ['aug2']:
            test_part_transform = transforms.Compose([
                transforms.Resize(self.raw_shape[-2:]),
            ])

            test_transform = transforms.Compose([
                #transforms.Lambda(remove_red_lines),
                #transforms.Lambda(remove_glare),
                transforms.Grayscale(),
                #transforms.Lambda(CLAHE()),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
            transform = transforms.Compose([
                transforms.Resize(self.raw_shape[-2:]),
                #transforms.Lambda(remove_red_lines),
                #transforms.Lambda(remove_glare),
                transforms.Grayscale(),
                transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
                #transforms.Lambda(CLAHE()),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(self.shape[-1]),
                transforms.ToTensor(),
                transforms.Lambda(AWGN(0.001)),
                transforms.Normalize(self.mean, self.std)
            ])
        elif preproc in ['aug3']:
            test_part_transform = transforms.Compose([
                transforms.Resize(self.raw_shape[-2:]),
            ])

            test_transform = transforms.Compose([
                #transforms.Lambda(remove_red_lines),
                # transforms.Lambda(remove_glare),
                transforms.Grayscale(),
                transforms.Lambda(CLAHE()),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])

            transform = transforms.Compose([
                transforms.Resize(self.raw_shape[-2:]),
                # transforms.Lambda(remove_red_lines),
                # transforms.Lambda(remove_glare),
                transforms.Grayscale(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomCrop(self.shape[-1]),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.Lambda(CLAHE()),
                transforms.ToTensor(),
                transforms.RandomApply([
                    transforms.GaussianBlur(3),
                    transforms.RandomErasing(value=1),
                    transforms.RandomAffine(degrees=60, scale=(0.9, 1.1), translate=(0, 0.4))
                ], p=0.3),
                transforms.Lambda(AWGN(0.01)),
                transforms.Normalize(self.mean, self.std)
            ])
        target_transform = transforms.Lambda(
            TargetTransFunctor(self.anomalous_label,
                               self.outlier_classes,
                               self.nominal_label)
        )
        if online_supervision:
            all_transform = MultiCompose([OnlineSuperviser(self, supervise_mode, noise_mode, oe_limit), *all_transform])
        else:
            all_transform = MultiCompose(all_transform)

        train_set = MYKeen(root=self.root, train=True,
                           transform=transform, target_transform=target_transform, all_transform=all_transform,
                           inv_mean=-self.mean/self.std, inv_std=1/self.std)
        train_set.targets = torch.tensor(train_set.targets)

        self._generate_artificial_anomalies_train_set(
            supervise_mode if not online_supervision else 'unsupervised', noise_mode,
            oe_limit, train_set, normal_class
        )

        self._test_set = MYKeen(root=self.root, train=False,
                                transform=test_transform, target_transform=target_transform,
                                test_part_transform=test_part_transform,
                                inv_mean=-self.mean/self.std, inv_std=1/self.std)


class MYKeen(ImageFolder):
    """ Fashion-MNIST dataset extension, s.t. target_transform and online superviser is applied """
    def __init__(self, root, logger=None, train=True, transform=None, target_transform=None, all_transform=None,
                 test_part_transform = None,
                 inv_mean=None, inv_std=None):
        if train:
            root = pt.join(root, 'train')
        else:
            root = pt.join(root, 'test')
        super(MYKeen, self).__init__(root=root, transform=transform, target_transform=target_transform)
        if 'Normal operating state' in self.class_to_idx and self.class_to_idx['Normal operating state'] or \
                'Flooding' in self.class_to_idx and self.class_to_idx['Flooding'] == 0:
            self.samples = [(x[0], 1 - x[1]) for x in self.samples]
            self.targets = [1 - t for t in self.targets]
            self.class_to_idx['Flooding'] = 1
            self.class_to_idx['Normal operating state'] = 0

        self.all_transform = all_transform
        self.test_part_transform = test_part_transform

        self.inv_mean = inv_mean
        self.inv_std = inv_std
        self.train = train

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # apply online superviser, if available
        if self.all_transform is not None:
            img = to_tensor(img)
            img, _, target = self.all_transform((img, None, target))
            img = img.sub(img.min()).div(img.max() - img.min()).mul(255).byte() if img.dtype != torch.uint8 else img
            img = to_pil_image(img)

        if self.test_part_transform is not None:
            img = self.test_part_transform(img)
            img_original = to_tensor(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.train or self.test_part_transform is None:
            return img, target
        else:
            return img, target, img_original

    # def inverse_normalize(self, imgs):
    #     if self.inv_mean is not None and self.inv_std is not None:
    #         return normalize(imgs, mean=self.inv_mean, std=self.inv_std)
    #     else:
    #         return imgs


