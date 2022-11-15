import torch

from EncoderDecoder import *
from Dataloader_1 import *
from matplotlib import pyplot as plt
from utils import *
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


test_net = net
model_param = './weights/model-35.pth'#35号 效果好
if model_param:
    test_net.load_state_dict(torch.load(model_param))

#weights_dict = torch.load('./weights/model-111.pth')
#print(weights_dict)
acc = 0
num = len(dataloader['val'].dataset)
acc,mse = evaluate(test_net, dataloader['val'], device=torch.device('cpu'))
# print(acc)
plot_num = min(4, dataloader['train'].batch_size)
# mse = 0
for x,y,z in dataloader['train']:
    # print(z)
    out = test_net(x)
    # acc += accuracy(out[1], z)
    # print(out[1])

    _,predi = torch.max(out[1],dim=1)
    # mse += nn.MSELoss()(out[0], y).item()
    #print(mse)
    plot_num = min(plot_num, len(y))
    for i in range(plot_num):
        plt.subplot(2, plot_num, i + 1)
        plt.xlabel(f'denoise{predi[i]}')
        plt.imshow(out[0][i].permute(1, 2, 0).detach().numpy())
        plt.subplot(2, plot_num, plot_num+i+1)
        plt.xlabel(z[i])
        plt.imshow(y[i].permute(1, 2, 0))
    plt.show()
# mean_mse = mse / len(dataloader['val'])#len means the number of iter
# print(f'MEAN_MSELoss:{mean_mse}')

    # print(nn.MSELoss()(out[0], y).item())
# accrate = acc / num


