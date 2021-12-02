
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import glob
from IPython.display import display
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from tqdm import tqdm_notebook
import random
import timm

import pandas as pd
import numpy as np


res_img_dir = r"C:\Users\THOMA\Documents\Computer Science and Math\EIT UCA\Deep Learning\kaggle\food-11\evaluation"
class testDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir):
        super().__init__()
        self.img_dir = img_dir
        # use glob to get all image names
        self.img_names = [x.rsplit("\\")[-1] for x in glob.glob(img_dir + "\*") ]
        
        # PyTorch transforms
        self.transform = transforms.Compose([#transforms.RandomHorizontalFlip(p=0.5),
                                             #transforms.RandomRotation(180),
                                             #transforms.ColorJitter(),
                                             # transforms.RandomRotation(45),
                                             # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                             # transforms.Resize((448, 448)),
                                               transforms.Resize((224, 224)),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            ])
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, i):
        return self._read_img(i)

    def _read_img(self, i):
        img = Image.open(self.img_dir + "\\" + self.img_names[i])
        return self.transform(img), self.img_names[i]

batch_size = 32
net = models.inception_v3(pretrained=False,aux_logits=False)
net.fc = nn.Linear(2048, 11)
net.load_state_dict(torch.load("inceptionV3_epoch_16_model.pth"))


net.cuda()
res_test_dataset = testDataset(res_img_dir)
test_dl = torch.utils.data.DataLoader(res_test_dataset, batch_size=batch_size)
net.eval()

res_Id = []
res_label = []


for i in range(len(res_test_dataset)):
    with torch.no_grad():
        x = torch.cat([res_test_dataset[i][0].unsqueeze(0) for _ in range(20)], 0)
        filename = res_test_dataset[i][1]
        x = x.cuda()
        y = net(x)
        res_Id.append( filename.split(".")[0])
        res_label.append((y.max(1)[1]).mode()[0].item())


df = pd.DataFrame()
df["Id"] = pd.Series(res_Id)
df["Category"] = pd.Series(res_label)
df.to_csv("submission_pandas.csv")