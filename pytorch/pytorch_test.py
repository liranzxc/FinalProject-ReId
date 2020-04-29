import os
import pickle
import sys
import time
from collections import deque

import torch
import numpy as np
import torch.nn as nn
import torchvision
from pair_dataset import PairDataset
from pytorch_metric_learning.losses import CosFaceLoss, ContrastiveLoss
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

# load folder parent
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 16, 5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 64, 7, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Sequential(
            nn.Linear(39168, 1),
            nn.Sigmoid()  # 0 ~ 1
        )

    def forward(self, x, y):
        x = self.cnn(x)
        x = x.view(-1, 64)
        y = self.cnn(y)
        y = y.view(-1, 64)
        B = torch.cat((x, y))
        B = B.view(-1)
        out = self.out(B)
        return out


def imshow(imgs: [], label):
    plt.subplot(1, 2, 1)
    img = imgs[0] / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    t1 = np.transpose(npimg, (1, 2, 0))

    plt.imshow(t1)

    plt.subplot(1, 2, 2)
    img = imgs[1] / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    t2 = np.transpose(npimg, (1, 2, 0))
    plt.imshow(t2)

    plt.xlabel("match ?  " + str(label))
    plt.show()


if __name__ == "__main__":
    path = '../dataset/re-id/campus/*'

    # create transform
    transform = transforms.Compose([
        transforms.Resize((160, 60)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = [0, 1]
    batch_size = 1
    train_dataset = PairDataset(path, mode="train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = PairDataset(path, mode="test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = PairDataset(path, mode="valid", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # get some random training images
    #
    # for batch_id, (imgs1, imgs2, labels) in enumerate(train_loader, 1):
    #     for img1,img2,label in zip(imgs1,imgs2,labels):
    #         # show images
    #         x = torchvision.utils.make_grid(img1)
    #         y = torchvision.utils.make_grid(img2)
    #         labelX = int(label.numpy())
    #         imshow([x, y],labelX)
    #         print(labelX)

    net = Siamese()
    net.train()

    cuda = True
    if cuda:
        net.cuda()

    lr = 1e-3  # Learning rate
    epoch_num = 2
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    stuck = 0
    loss_log = []

    criterion = nn.MSELoss()
    stop = False
    for ep in range(epoch_num):  # epochs loop
        if stop:
            break
        running_loss = 0.0
        for batch_id, (img1, img2, yi) in enumerate(train_loader, 0):

            optimizer.zero_grad()

            if cuda:
                img1, img2, yi = Variable(img1.cuda()), Variable(img2.cuda()), Variable(yi.cuda())
            else:
                img1, img2, yi = Variable(img1), Variable(img2), Variable(yi)

            pred = net.forward(img1, img2)  # 0 ~ 1

            # yi if y = 1 are the same pair
            # yi if y = 0 are not the same pair
            loss = criterion(yi, pred)

            # print("pred", pred)
            # print("yi actual", yi)
            # print("loss", loss)
            # print("*"*70)
            # Backward pass and updates
            loss.backward()  # calculate the gradients (backpropagation)
            optimizer.step()  # update the weights
            running_loss += loss.item()

            if batch_id % 500 == 499:  # print every 500 mini-batches
                print('[%d, %5d] train loss: %.3f' %
                      (ep + 1, batch_id + 1, running_loss / 500))

                if len(loss_log) > 0:
                    if (running_loss / 500) == loss_log[-1]:
                        stuck = stuck + 1
                    else:
                        stuck = 0

                loss_log.append(running_loss / 500)

                if stuck > 4 :
                    stop = True
                    print("stuck")
                    break

              #  if (running_loss / 500) < 0.001 and running_loss != 0.0:  # thehold
                if batch_id % 4000 == 3999:  # print every 500 mini-batches
                    right = 0
                    error = 0
                    n = 0
                    for _, (img1_val, img2_val, yi_val) in enumerate(val_loader, 1):
                        if cuda:
                            img1_val, img2_val, yi_val = Variable(img1_val.cuda()), Variable(img2_val.cuda()), Variable(
                                yi_val.cuda())
                        else:
                            img1_val, img2_val, yi_val = Variable(img1_val), Variable(img2_val), Variable(yi_val)

                        pred = net.forward(img1_val, img2_val).cpu().detach().numpy()
                        yi_val = yi_val.item()
                        if np.round_(pred) == yi_val:
                            right += 1
                        else:
                            error += 1
                        n += 1

                    print("acc {}".format(right / n))
                    print("error {}".format(error / n))
                    print("*" * 70)

                    # stop = True
                    # break
                running_loss = 0.0

            # if batch_id % 8000 == 7999:
            #     # check acc
            #     right = 0
            #     error = 0
            #     n = 0
            #     for _, (img1_val, img2_val, yi_val) in enumerate(val_loader, 1):
            #         if cuda:
            #             img1_val, img2_val, yi_val = Variable(img1_val.cuda()), Variable(img2_val.cuda()), Variable(
            #                 yi_val.cuda())
            #         else:
            #             img1_val, img2_val, yi_val = Variable(img1_val), Variable(img2_val), Variable(yi_val)
            #
            #         pred = net.forward(img1_val, img2_val).cpu().detach().numpy()
            #         yi_val = yi_val.cpu().detach().numpy()
            #         yi_val = yi_val[0]
            #         if np.round_(pred) == yi_val:
            #             right += 1
            #         else:
            #             error += 1
            #         n += 1
            #
            #     print("acc {}".format(right / n))
            #     print("error {}".format(error / n))
            #     print("*" * 70)

    torch.save(net.state_dict(), '../reId/siamese_60w_160h.pth')

    plt.plot(loss_log)
    plt.title('Loss')
    plt.show()
