from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
import os.path as pt
import torch
import numpy as np
from fcdd.datasets.preprocessing import local_contrast_normalization_func, local_contrast_normalization_func2, \
    MultiCompose, AWGN, BlackCenter, TargetTransFunctor, remove_red_lines, CLAHE, remove_glare


def ceil(x: float):
    return int(np.ceil(x))


class MYKeen(ImageFolder):
    def __init__(self, root, logger=None, train=True, transform=None, target_transform=None, all_transform=None):
        if train:
            root = pt.join(root, 'train_oe')
        else:
            root = pt.join(root, 'test')
        super(MYKeen, self).__init__(root=root, transform=transform, target_transform=target_transform)
        if self.class_to_idx['Flooding'] == 0:
            self.samples = [(x[0], 1 - x[1]) for x in self.samples]
            self.targets = [1 - t for t in self.targets]
            self.class_to_idx['Flooding'] = 1
            #self.class_to_idx['Normal operating state'] = 0

    """ Reimplements get_item to transform tensor input to pil image before applying transformation. """
    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class OEKeen(MYKeen):
    keen_classes = ['flooding']

    def __init__(self, size: torch.Size, root: str = None, train: bool = True, limit_var: int = 20):
        """
        Outlier Exposure dataset for KEEN.
        :param size: size of the samples in n x c x h x w, samples will be resized to h x w. If n is larger than the
            number of samples available in KEEN, dataset will be enlarged by repetitions to fit n.
            This is important as exactly n images are extracted per iteration of the data_loader.
            For online supervision n should be set to 1 because only one sample is extracted at a time.
        :param root: root directory where data is found or is to be downloaded to.
        :param train: whether to use training or test samples.
        :param limit_var: limits the number of different samples, i.e. randomly chooses limit_var many samples
            from all available ones to be the training data.
        """
        assert len(size) == 4 #and size[2] == size[3]
        assert size[1] in [1, 3]
        if size == 1:
            transform = transforms.Compose([
                transforms.Resize((size[2], size[3])),
                transforms.ToTensor()
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((size[2], size[3])),
                transforms.ToTensor()
            ])
        super().__init__(root,
                         # train,
                         transform=transform)
        self.size = size
        self.targets = torch.from_numpy(np.asarray(self.targets))
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        if limit_var is not None and limit_var < len(self):
            picks = np.random.choice(np.arange(len(self.samples)), size=limit_var, replace=False)
            self.samples = self.samples[picks]
            self.targets = self.targets[picks]
        if limit_var is not None and limit_var > len(self):
            print(
                'KEEN shall be limited to {} samples, but KEEN contains only {} samples, thus using all.'
                .format(limit_var, len(self))
            )
        # if len(self) < size[0]:
        #     rep = ceil(size[0] / len(self))
        #     old = len(self)
        #     self.data = self.data.repeat(rep, 1, 1, 1)
        #     self.targets = self.targets.repeat(rep)
        #     if rep != size[0] / old:
        #         import warnings
        #         warnings.warn(
        #             'OECifar100 has been limited to {} samples. '
        #             'Due to the requested size of {}, the dataset will be enlarged. '
        #             'But {} repetitions will make some samples appear more often than others in the dataset, '
        #             'because the final size after repetitions is {}, which is cut to {}'
        #             .format(limit_var, size[0], rep, len(self), size[0])
        #         )

    def data_loader(self):
        return DataLoader(dataset=self, batch_size=self.size[0], shuffle=True, num_workers=0)

    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        sample = sample.mul(255).byte()

        return sample
