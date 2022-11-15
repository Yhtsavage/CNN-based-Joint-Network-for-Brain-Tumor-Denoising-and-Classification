import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import torchvision.utils
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
import numpy as np
import os
from utils import *

def Gaussian_Noise(img):
    img_noise = img.numpy().copy()
    s1, s2, s3 = img.shape
    means = torch.zeros(s1, s2, s3)
    std = torch.ones(s1, s2, s3)
    sigma = 0.5
    std = std * sigma
    noise = torch.normal(means, std)
    noise = noise.numpy()
    img_noise = img_noise + noise
    img_noise = torch.from_numpy(img_noise)  # 转噪声图像为tensor
    return img_noise

def salt_pepper(img):
    SNR = 0.9
    #plt.imshow(img.permute(1, 2, 0).detach().numpy())
    #plt.show()
    img_ = img.numpy().copy()# 不加copy 噪声会加在img上，这样子的话 ground true也是有噪图
    c, h, w = img_.shape
    mask = np.random.choice((0, 1, 2), size=(1, h, w), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
    mask = np.repeat(mask, c, axis=0)     # 按channel 复制到 与img具有相同的shape
    img_[mask == 1] = 255    # 盐噪声
    img_[mask == 2] = 0      # 椒噪声
    img_noise = torch.from_numpy(img_)
    #plt.imshow(img_noise.permute(1, 2, 0))
    #plt.show()
    return img_noise

class Tumor(Dataset):
    def __init__(self, img_path, data, transform, noise):
        self.transform = transform
        self.img_name = data.Image.tolist() #img1，img2，img3
        self.label = data.Class.tolist()
        self.img_folder = img_path
        self.noise = noise
    def __len__(self):
        return len(self.label)
    def __getitem__(self, item):
        path = os.path.join(self.img_folder,str(self.img_name[item]) + '.jpg')
        img = Image.open(path)
        img = self.transform(img)
        label = self.label[item]
        img_nosie = self.noise(img)
        return img_nosie, img, label

path_csv = './dataset/Brain Tumor.csv'
path_img = './dataset/Brain Tumor/Brain Tumor'
data = pd.read_csv(path_csv, usecols=[0, 1])#取img，和class
#print(data)
train_val, test = train_test_split(data, test_size=0.1, shuffle=True, random_state= 45)
train, val = train_test_split(train_val, test_size=0.1, shuffle=True, random_state= 45)
data_list = ['train', 'val', 'test', 'train_val']
Nosie = Gaussian_Noise
batch_size = 16
img_data = {
    'train' : train,
    'val' : val,
    'test' : test,
    'train_val' : train_val
    }
#print(data.Class.tolist()[10]) #data.loc[0][1]
#print(data.Image.tolist()[10]) #data.loc[0][0]
#print(len(train.Class.tolist()))
#print(len(train.Image.tolist()))
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
data_transform = {
    'train' : transforms.Compose([transforms.Resize(244),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor()
                                  #transforms.Normalize(mean, std)
                                  ]),
    'val' : transforms.Compose([transforms.CenterCrop(224),
                                transforms.ToTensor()]),
    'test' : transforms.Compose([transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean, std)]),
    'train_val' : transforms.Compose([transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean, std)]),
}
dataset = { x : Tumor(path_img, img_data[x], transform=data_transform[x], noise=Nosie) for x in data_list}
dataloader = {x : DataLoader(dataset[x], batch_size, shuffle=True) for x in data_list}

if __name__ == '__main__':
    plot_data_loader_img(data_loader=dataloader['train'])


