from torch.utils.data import Dataset
import numpy as np


class PairDataset(Dataset):
    """Pair dataset."""

    def __init__(self, dataManager, train=True, transform=None):
        self.transform = transform
        self.train = train
        self.train_data, self.train_labels = self.createDb(dataManager.train_loader,Train=True)
        self.test_data, self.test_labels = self.createDb(dataManager.test_loader,Train=False)

    def setTrain(self, train):
        self.train = train

    def createDb(self, dataLoader,Train=True):
        dataset = []
        labels = []
        pids = {}
        for batch in dataLoader:
            if not Train:
                print(batch)

            images, ids, cids, paths = batch
            for image, _id, cid, path in zip(images, ids, cids, paths):
                npid = int(_id.numpy())
                if pids.get(npid, None) is None:
                    pids[npid] = [(image, _id, cid, path)]
                else:
                    pids[npid].append((image, _id, cid, path))

        # create same images
        for _id in pids.keys():
            imgs, label = self.createPairImages(pids[_id], pids[_id], Same=True)
            dataset.append(imgs)
            labels.append(label)

        # create not same images
        for _id in pids.keys():
            all_ids = list(pids.keys())
            all_ids.remove(_id)
            select_id = np.random.choice(all_ids)
            imgs, label = self.createPairImages(pids[_id], pids[select_id], Same=False)
            dataset.append(imgs)
            labels.append(label)

        return dataset, labels

    def createPairImages(self, p1, p2, Same=True):
        randImg = np.random.choice(p1[0])
        randImg2 = np.random.choice(p2[0])
        label = 1 if Same else 0
        return (randImg, randImg2), label

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        if self.train:
            imgs, target = self.train_data[index], self.train_labels[index]
        else:
            imgs, target = self.test_data[index], self.test_labels[index]

        img_ar = []
        for i in range(len(imgs)):
            img = Image.fromarray(imgs[i].numpy(), mode='L')
            if self.transform is not None:
                img = self.transform(img)
            img_ar.append(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_ar, target
