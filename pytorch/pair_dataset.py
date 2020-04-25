import itertools

import torch
from torch.utils.data import Dataset
import numpy as np
from glob import glob
from PIL import Image
from torchvision.transforms import transforms


class PairDataset(Dataset):
    """Pair dataset."""

    def __init__(self, path, mode="train", transform=None):
        self.transform = transform
        self.mode = mode
        self.same_labels = 0
        self.odd_labels = 0
        self.path = path
        self.train_data, self.train_label, self.test_data, self.test_label, self.val_data, self.val_label = self.create_db()

    def change_mode(self, mode):
        self.mode = mode

    def create_db(self):
        ids = {}
        paths = glob(self.path)
        paths.sort()

        # reduce by id
        for path in paths:
            name = path.split("/")[-1]
            _id = name[:4]
            if ids.get(_id, None) is None:
                ids[_id] = [path]
            else:
                ids[_id].append(path)

        # create pairs
        pairs_same = []
        labels_same = []
        pairs_odd = []
        labels_odd = []
        # create same pairs
        for key in ids.keys():
            images = ids[key]
            combinations = list(itertools.combinations(images, 2))
            pairs_same.extend(combinations)
            labels_same.extend([1] * len(combinations))
            self.same_labels = self.same_labels + len(combinations)

        # create different pairs
        for key in ids.keys():
            list_ids = list(ids.keys())
            list_ids.remove(key)
            selected_index = np.random.choice(list_ids)
            combinations = list(itertools.product(ids[selected_index], ids[key]))
            pairs_odd.extend(combinations)
            labels_odd.extend([0] * len(combinations))
            self.odd_labels = self.odd_labels + len(combinations)

            if self.odd_labels > self.same_labels:
                break

        # split the data 70% train , 10% valid , 20% test

        train_pair = pairs_same[: int(len(pairs_same) * 0.7)] + pairs_odd[: int(len(pairs_odd) * 0.7)]

        train_label = labels_same[: int(len(labels_same) * 0.7)] + labels_odd[: int(len(labels_odd) * 0.7)]

        val_pair = pairs_same[int(len(pairs_same) * 0.7): int(len(pairs_same) * 0.8)] \
                   + pairs_odd[int(len(pairs_odd) * 0.7): int(len(pairs_odd) * 0.8)]
        val_label = labels_same[int(len(labels_same) * 0.7): int(len(labels_same) * 0.8)] \
                    + labels_odd[int(len(labels_odd) * 0.7): int(len(labels_odd) * 0.8)]

        test_pair = pairs_same[int(len(pairs_same) * 0.8):] + pairs_odd[int(len(pairs_odd) * 0.8):]
        test_label = labels_same[int(len(labels_same) * 0.8):] + labels_odd[int(len(labels_odd) * 0.8):]

        return train_pair, train_label, test_pair, test_label, val_pair, val_label

    def details(self):
        print("# of pair train", len(self.train_data))
        print("# of labels train", len(self.train_label))

        print("# of pair test", len(self.test_data))
        print("# of labels test", len(self.test_label))

        print("# of pair val", len(self.val_data))
        print("# of labels val", len(self.val_label))

        print("# of same pairs in all db", self.same_labels)
        print("# of diff pairs in all db", self.odd_labels)

    def __len__(self):
        if self.mode is "train":
            return len(self.train_data)
        elif self.mode is "test":
            return len(self.test_data)
        else:
            return len(self.val_data)

    def __getitem__(self, index):
        if self.mode is "train":
            imgs, label = self.train_data[index], self.train_label[index]
        elif self.mode is "test":
            imgs, label = self.test_data[index], self.test_label[index]
        else:
            imgs, label = self.val_data[index], self.val_label[index]

        imgs_arr = []

        for path in imgs:
            img = Image.open(path)
            if self.transform is not None:
                img = self.transform(img)
                pil_to_tensor = img
            else:
                pil_to_tensor = transforms.ToTensor()(img).unsqueeze_(0)

            imgs_arr.append(pil_to_tensor)

        return imgs_arr[0], imgs_arr[1], torch.from_numpy(np.array([label], dtype=np.float32))
