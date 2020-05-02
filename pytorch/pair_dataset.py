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
        self.path = path

        self.train_data = []
        self.train_label = []

        self.test_data = []
        self.test_label = []

        self.create_db()

    def change_mode(self, mode):
        self.mode = mode

    def create_db(self):
        ids = {}
        paths = glob(self.path)
        paths.sort()
        # reduce by id
        for path in paths:
            name = path.split("/")[-1][:4]
            _id = name.lstrip("0")
            if ids.get(_id, None) is None:
                ids[_id] = [path]
            else:
                ids[_id].append(path)

        number_of_ids = len(ids.keys()) - 1
        # train data
        for key in ids.keys():
            for i in range(20):
                rnd_cls = np.random.randint(1, number_of_ids)  # choose random class that is not the same class
                if str(rnd_cls) == key:
                    rnd_cls = rnd_cls + 1

                rnd_cls = str(rnd_cls)
                images_person_A = ids[key]
                images_person_B = ids[rnd_cls]

                image_A_1_index = np.random.randint(1, len(images_person_A) - 1)
                image_A_2_index = np.random.randint(1, len(images_person_A) - 1)
                image_B_1_index = np.random.randint(1, len(images_person_B) - 1)

                img_a_1_path = images_person_A[image_A_1_index]
                img_a_2_path = images_person_A[image_A_2_index]
                img_b_1_path = images_person_B[image_B_1_index]

                data = [img_a_1_path, img_a_2_path]
                self.train_data.append(data)
                self.train_label.append(np.array([0], dtype=np.float32))  # similarity

                data = [img_b_1_path, img_a_2_path]
                self.train_data.append(data)
                self.train_label.append(np.array([1], dtype=np.float32))

        # test data
        for key in ids.keys():
            for i in range(20):
                rnd_cls = np.random.randint(1, number_of_ids)  # choose random class that is not the same class
                if str(rnd_cls) == key:
                    rnd_cls = rnd_cls + 1

                rnd_cls = str(rnd_cls)
                images_person_A = ids[key]
                images_person_B = ids[rnd_cls]

                image_A_1_index = np.random.randint(1, len(images_person_A) - 1)
                image_A_2_index = np.random.randint(1, len(images_person_A) - 1)
                image_B_1_index = np.random.randint(1, len(images_person_B) - 1)

                img_a_1_path = images_person_A[image_A_1_index]
                img_a_2_path = images_person_A[image_A_2_index]
                img_b_1_path = images_person_B[image_B_1_index]

                data = [img_a_1_path, img_a_2_path]
                self.test_data.append(data)
                self.test_label.append(np.array([0], dtype=np.float32))  # similarity

                data = [img_b_1_path, img_a_2_path]
                self.test_data.append(data)
                self.test_label.append(np.array([1], dtype=np.float32))

    def __len__(self):
        if self.mode is "train":
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        if self.mode is "train":
            imgs, label = self.train_data[index], self.train_label[index]
        else:
            imgs, label = self.test_data[index], self.test_label[index]

        imgs_arr = []
        for path in imgs:
            img = Image.open(path)
            if self.transform is not None:
                img = self.transform(img)
                pil_to_tensor = img
            else:
                pil_to_tensor = transforms.ToTensor()(img).unsqueeze_(0)

            imgs_arr.append(pil_to_tensor)

        return imgs_arr, label
