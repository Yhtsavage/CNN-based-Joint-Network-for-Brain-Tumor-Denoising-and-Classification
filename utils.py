from tqdm import tqdm
import sys
import torch
import os
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def lossfunction(net_out, ground_true, label, loss_fc, loss_denoise, ratio):
    loss = ratio[0]*loss_denoise(net_out[0],ground_true)+ratio[1]*loss_fc(net_out[1],label)
    return loss

def train_one_epoch(model, optimizer, data_loader, device, epoch, ratio):
    model.train()
    model.to(device)
    loss_fc = nn.CrossEntropyLoss()
    loss_denoise = nn.MSELoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()
    data_loader = tqdm(data_loader, file=sys.stdout)
    for iter, data in enumerate(data_loader):
        inputs = data[0].to(device)
        gture = data[1].to(device)
        label = data[2].to(device)
        net_out = model(inputs)
        loss = lossfunction(net_out, gture, label, loss_fc, loss_denoise, ratio)
        loss.backward()
        mean_loss = (mean_loss * iter + loss.detach()) / (iter + 1) #前+现在/经历的batch数
        data_loader.desc = f"[epoch{epoch}] mean_loss{round(mean_loss.item(), 5)}"

        if not torch.isfinite(loss):
            print("ending training", loss)
            sys.exit(1)
        optimizer.step()
        optimizer.zero_grad()
    return mean_loss.item()

@torch.no_grad()#decorator
def evaluate(model, data_loader, device):
    model.eval()
    #model.to(device)
    sum_num = torch.zeros(1).to(device)
    mse_sum = torch.zeros(1).to(device)
    #cosine_simila = torch.zeros(1).to(device)
    num_samples = len(data_loader.dataset)#所有的样本数
    data_loader = tqdm(data_loader, desc="testing...", file=sys.stdout)
    for iter, data in enumerate(data_loader):
        inputs = data[0].to(device)
        gtrue = data[1].to(device)
        labels = data[2].to(device)
        denoise,pred = model(inputs)
        sum_num += accuracy(pred, labels)
        mse_sum = (mse_sum * iter + nn.MSELoss()(denoise, gtrue))/(iter + 1)#already average for batch size
    acc = sum_num / num_samples
    mse_loss = mse_sum
    return acc, mse_loss

def plot_data_loader_img(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 3)
    for data in data_loader:
        imgs_noise, images, labels = data
        for i in range(0, plot_num):
            image = images[i].permute(1, 2, 0)
            image_noise = imgs_noise[i].permute(1, 2, 0)
            #image = (image * [0.299, 0.244, 0.255] + [0.485, 0.456, 0.406]) * 255 #反Normalization
            label = labels[i].item()
            plt.subplot(1, 2 * plot_num, i+1)
            plt.xlabel(label)
            plt.xticks([]),plt.yticks([])
            plt.imshow(image)
            plt.subplot(1, 2 * plot_num, plot_num+i+1)
            plt.xlabel(f'noise{label}')
            plt.xticks([]),plt.yticks([])
            plt.imshow(image_noise)
        plt.show()

def denoise_image_board(data_loader, model, device, epoch, path):
    model.eval()
    model.to(device)
    batch_size = data_loader.batch_size
    #plot_num = min(batch_size, 4)
    for data in data_loader:
        plot_num = min(len(data[0]), 2)
        noise_images = data[0].to(device)
        images = data[1].to(device)
        label = data[2].to(device)
        denoise_images,_ = model(noise_images)
        #fig = plt.figure(epoch+1, figsize=(540,540))
        #画图部分 如果画多张图就要循环
        for i in range(plot_num):
            denoise = denoise_images[i].permute(1, 2, 0).detach().numpy()
            noise_image = noise_images[i].permute(1, 2, 0).detach().numpy()
            plt.subplot(1, plot_num*2, i+1)
            plt.xlabel('Denoise')
            plt.xticks([]),plt.yticks([])
            plt.imshow(denoise)
            plt.subplot(1, plot_num*2, plot_num+i+1)
            plt.xlabel('Noise image')
            plt.xticks([]),plt.yticks([])
            plt.imshow(noise_image)
        plt.suptitle(f'epoch = {epoch}')
        plt.savefig(path)
        plt.show()
        break
            #return fig









