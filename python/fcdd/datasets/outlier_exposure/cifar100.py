from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os.path as pt
import torch
import numpy as np


def ceil(x):
    return int(np.ceil(x))


class MYCIFAR100(CIFAR100):
    def __getitem__(self, index):
        """
        Reimplements getitem to transform tensor to pil image.
        """
        img, target = self.data[index], self.targets[index]

        img = transforms.ToPILImage()(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class OECifar100(MYCIFAR100):
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def __init__(self, size, root=None, train=True, limit_var=20):  # split = Train
        assert len(size) == 4 and size[2] == size[3]
        assert size[1] in [1, 3]
        root = pt.join(root, 'cifar100', )
        transform = transforms.Compose([
            transforms.Resize((size[2], size[3])),
            transforms.Grayscale() if size[1] == 1 else transforms.Lambda(lambda x: x),
            transforms.ToTensor()
        ])
        super().__init__(root, train, transform=transform, download=True)
        self.size = size
        self.targets = torch.from_numpy(np.asarray(self.targets))
        self.data = torch.from_numpy(self.data).transpose(1, 3).transpose(2, 3)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        if limit_var is not None and limit_var < len(self):
            picks = np.random.choice(np.arange(self.data.size(0)), size=limit_var, replace=False)
            self.data = self.data[picks]
            self.targets = self.targets[picks]
        if limit_var is not None and limit_var > len(self):
            print(
                'OECifar100 shall be limited to {} samples, but Cifar100 contains only {} samples, thus using all.'
                .format(limit_var, len(self))
            )
        if len(self) < size[0]:
            rep = ceil(size[0] / len(self))
            old = len(self)
            self.data = self.data.repeat(rep, 1, 1, 1)
            self.targets = self.targets.repeat(rep)
            if rep != size[0] / old:
                import warnings
                warnings.warn(
                    'OECifar100 has been limited to {} samples. '
                    'Due to the requested size of {}, the dataset will be enlarged. ' 
                    'But {} repetitions will make some samples appear more often than others in the dataset, '
                    'because the final size after repetitions is {}, which is cut to {}'
                    .format(limit_var, size[0], rep, len(self), size[0])
                )

    def data_loader(self):
        return DataLoader(dataset=self, batch_size=self.size[0], shuffle=True, num_workers=0)

    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        sample = sample.mul(255).byte()

        return sample