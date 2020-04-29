import torch
from PIL import Image
from torchvision.transforms import transforms
import torchvision
from pytorch.pytorch_test import Siamese
import matplotlib.pyplot as plt
path1 = "../dataset/re-id/campus/0001001.png"
path2 = "../dataset/re-id/campus/0001002.png"
path3 = "../dataset/re-id/campus/0011003.png"

transform = transforms.Compose([
    transforms.Resize((160, 60)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

img1 = Image.open(path1)
img2 = Image.open(path2)

img3 = Image.open(path3)

pil_to_tensor1 = transform(img1)
pil_to_tensor2 = transform(img2)
pil_to_tensor3 = transform(img3)

pil_to_tensor1 = pil_to_tensor1.view(1, 3, 160, 60)
pil_to_tensor2 = pil_to_tensor2.view(1, 3, 160, 60)
pil_to_tensor3 = pil_to_tensor3.view(1, 3, 160, 60)

pathmodel = "../reId/siamese_60w_160h.pth"
model = Siamese()

model.eval()

checkpoint = torch.load(pathmodel)
model.load_state_dict(checkpoint)

pred1 = model.forward(pil_to_tensor1, pil_to_tensor1)

pred2 = model.forward(pil_to_tensor2, pil_to_tensor2)

pred3 = model.forward(pil_to_tensor3, pil_to_tensor3)

pils = torch.cat([pil_to_tensor1,pil_to_tensor2,pil_to_tensor3],dim=0)

print(pils.size())
grid = torchvision.utils.make_grid(pils,nrow=5)

plt.imshow(grid.permute(1,2,0))
plt.show()

print("1 vs 1 -",pred1)
print("2 vs 2 -",pred2)
print("3 vs 3 -",pred3)