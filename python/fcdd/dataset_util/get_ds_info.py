import argparse
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='Calculate info for ImageFolder dataset')
    my_parser.add_argument('-p', '--path',
                           type=str,
                           help='the path to dataset', required=True)
    args = my_parser.parse_args()

    transform = Compose([Resize((256, 256)), ToTensor()])

    dataset = ImageFolder(root=args.path, transform=transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    mean = 0.
    std = 0.
    nb_samples = 0

    for data, _ in tqdm(loader):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    print(mean, std)
