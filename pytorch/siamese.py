import os
import sys

import torch
from pytorch_metric_learning.losses import ContrastiveLoss
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import BCELoss, CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from pytorch.pair_dataset import PairDataset

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


class Siamese_V2(nn.Module):
    def __init__(self):
        super(Siamese_V2, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 10, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 7, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 128, 4, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 4, stride=1),
            nn.ReLU(inplace=True),

        )
        self.out = nn.Sequential(
            nn.Linear(47616, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        x = self.cnn(x)
        x = x.view(x.shape[0], -1)
        x = self.out(x)

        y = self.cnn(y)
        y = y.view(y.shape[0], -1)
        y = self.out(y)

        return x, y


def train(net, data_loader, number_epoch, lr, momentum=0.9, cuda=True):
    net.train()
    optimizer = optim.SGD(net.parameters(), momentum=momentum, lr=lr)
    loss_history = []
    criterion = ContrastiveLoss()

    if cuda:
        net.cuda()

    for epoch in range(number_epoch):
        for batch_idx, (imgs, target) in enumerate(data_loader):

            if cuda:
                for i in range(len(imgs)):
                    imgs[i] = Variable(imgs[i].cuda())

            p1, p2 = imgs[0], imgs[1]
            target = target.squeeze(0).type(torch.FloatTensor).cuda()

            optimizer.zero_grad()  # zero the gradient buffers
            output1, output2 = net.forward(p1, p2)
            loss = criterion(output1, output2, target)

            loss.backward()
            optimizer.step()  # Does the update

            if batch_idx % 5 == 0:
                print("batch_idx", batch_idx)
                print("loss", loss.item())
                loss_history.append(loss.item())

            if batch_idx % 100 == 99:
                break

    torch.save(net.state_dict(), '../reId/siamese_triplet.pth')

    plt.plot(loss_history)
    plt.title('Loss')
    plt.show()


if __name__ == "__main__":
    path = '../dataset/re-id/campus/*'

    # create transform
    transform = transforms.Compose([
        transforms.Resize((160, 60)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 1
    train_dataset = PairDataset(path, mode="train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = PairDataset(path, mode="test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    net = Siamese_V2()
    train(net, train_loader, 1, 0.01, momentum=0.9, cuda=True)
