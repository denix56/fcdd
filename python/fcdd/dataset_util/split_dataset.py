import argparse
import os
import numpy as np
import shutil

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='Split ImageFolder dataset in train, test and validation sets')
    my_parser.add_argument('--path',
                           type=str,
                           help='the path to dataset', required=True)
    my_parser.add_argument('--split',
                           type=float,
                           nargs=3,
                           help='train / test/ valid split', required=True)
    my_parser.add_argument('--seed',
                           type=int,
                           default=42,
                           help='random seed')

    args = my_parser.parse_args()
    path = args.path
    new_path = os.path.join(os.path.dirname(path), os.path.basename(path) + '_processed')

    if os.path.exists(new_path):
        shutil.rmtree(new_path)

    os.makedirs(os.path.join(new_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(new_path, 'test'), exist_ok=True)
    os.makedirs(os.path.join(new_path, 'valid'), exist_ok=True)

    for root, dirs, files in os.walk(path):
        for dir in dirs:
            os.makedirs(os.path.join(new_path, 'train', dir), exist_ok=True)
            os.makedirs(os.path.join(new_path, 'test', dir), exist_ok=True)
            os.makedirs(os.path.join(new_path, 'valid', dir), exist_ok=True)

        if files:
            indices = np.random.permutation(len(files))
            train_part = int(len(files)*args.split[0])
            test_part = train_part + int(len(files)*args.split[1])

            train_dir = os.path.join(new_path, 'train', os.path.basename(root))
            test_dir = os.path.join(new_path, 'test', os.path.basename(root))
            valid_dir = os.path.join(new_path, 'valid', os.path.basename(root))

            for i in indices[:train_part]:
                shutil.copy(os.path.join(root, files[i]), train_dir)
            for i in indices[train_part:test_part]:
                shutil.copy(os.path.join(root, files[i]), test_dir)
            for i in indices[test_part:]:
                shutil.copy(os.path.join(root, files[i]), valid_dir)

