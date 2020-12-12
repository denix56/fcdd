import argparse
import os
import numpy as np
import shutil
import re
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='Split ImageFolder dataset in train, test and validation sets')
    my_parser.add_argument('--path',
                           type=str,
                           help='the path to dataset', required=True)
    my_parser.add_argument('--train_size',
                           type=float,
                           help='train size', required=True)
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
    p = re.compile('([0-9]+)\\.jpg')

    print('New dataset path:', new_path)

    for root, dirs, files in os.walk(path):
        for dir in dirs:
            os.makedirs(os.path.join(new_path, 'train', dir), exist_ok=True)
            os.makedirs(os.path.join(new_path, 'test', dir), exist_ok=True)

        if files:
            ts_arr = np.array([int(p.search(f).group(1)) for f in files])
            ts_arr_idx = np.argsort(ts_arr)
            ts_groups = []
            split_i = 0
            for i, ts_idx in enumerate(ts_arr_idx[1:]):
                if ts_arr[ts_idx] - ts_arr[ts_arr_idx[i]] > 100:
                    ts_groups.append([files[idx] for idx in ts_arr_idx[split_i:i+1]])
                    split_i = i + 1

            ts_groups_train, ts_groups_test = train_test_split(ts_groups, train_size=args.train_size)

            train_dir = os.path.join(new_path, 'train', os.path.basename(root))
            test_dir = os.path.join(new_path, 'test', os.path.basename(root))

            for group in ts_groups_train:
                for f in group:
                    shutil.copy(os.path.join(root, f), train_dir)
            for group in ts_groups_test:
                for f in group:
                    shutil.copy(os.path.join(root, f), test_dir)

