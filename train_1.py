import os.path
from EncoderDecoder import *
from Dataloader_1 import *
import math
import argparse
from utils import *
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

def main(arg):
    device = torch.device(arg.device if torch.cuda.is_available() else "cpu")
    print(arg)
    print('Start Tensorboard ')
    if os.path.exists(arg.tensorboard_path) is False:
        os.makedirs(arg.tensorboard_path)
    tb_writer = SummaryWriter(log_dir=arg.tensorboard_path) #实例化
    if os.path.exists('./weights') is False:
        os.makedirs('./weights')
    if os.path.exists(arg.compare_path) is False:
        os.makedirs(arg.compare_path)
    if os.path.exists(arg.save_weights) is False:
        os.makedirs(arg.save_weights)
    model = net.to(device)
    #denoise_image_board(data_loader=dataloader['val'], model=model, device=torch.device("cpu"), epoch= 1)
    #init_img = torch.zeros((1, 3, 224, 224), device=device)
    #tb_writer.add_graph(model, init_img)
    if os.path.exists(arg.weights):
        weights_dict = torch.load(arg.weights, map_location=device)
        load_weights_dict = {k:v for k,v in weights_dict.items() #读取模型的值
                             if model.state_dict()[k].numel() == v.numel()}#需要的权重值数量相同的话即可"""
        model.load_state_dict(load_weights_dict)
    else:
        print('not using pretrain.')
    if arg.freeze:
        print('freeze not fc')
        for name, para in model.classif.named_parameters():
            if 'fc' not in name:
                para.requires_grad_(False)
    progra = [p for p in model.parameters() if p.requires_grad] #前向传递的一些参数
    optimizer = optim.SGD([{'params': progra, 'initial_lr': arg.lr}], lr=arg.lr, momentum=0.9, weight_decay=0.005)
    #optimizer = optim.SGD(progra, lr=arg.lr, momentum=0.9, weight_decay=0.005)
    lf = lambda x:((1 + math.cos(x * math.pi / arg.epochs)) / 2) * (1 - arg.lrf) + arg.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf, last_epoch=25)
    loss_ratio = arg.loss_ratio
    for epoch in range(arg.epochs):
        mean_loss = train_one_epoch(model, optimizer, data_loader=dataloader['train'], device=device, epoch=epoch, ratio=loss_ratio)
        scheduler.step()#updata the learning rate
        acc,mse = evaluate(model, data_loader=dataloader['val'], device=device)
        print(f'[epoch{epoch}] accuracy {round(acc.item(), 3)} mese{mse.item()}')
        if (epoch+1) % 10 == 0:
            img_path = f'{arg.compare_path}/img_{epoch+1}.png'
            denoise_image_board(data_loader=dataloader['val'], model=model, device=torch.device('cpu'), epoch= epoch+1, path=img_path)
        #model.to(device)
        #tb_writer.add_figure(tag='Comparison', figure=fig, global_step=epoch+1)
        tags = ['train_loss', 'accuracy', 'learning_rate', 'MSE','Compare']
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[3], mse, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]['lr'], epoch)
        #add fig
        #fig = plot_clas
        torch.save(model.state_dict(), f"{arg.save_weights}/model-{epoch}.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0085804620757699)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--loss_ratio', default=[0.98, 0.02])
    parser.add_argument('--weights', type=str, default="./weights/connection_0.98/model-25.pth")
    parser.add_argument('--freeze', type=bool, default=False)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--compare_path',type=str, default='./save_images/connection_0.98')
    parser.add_argument('--save_weights',type=str, default='./weights/connection_0.98')
    parser.add_argument('--tensorboard_path',type=str, default='./runs/TB/connection_0.98')
    opt = parser.parse_args()
    main(opt)