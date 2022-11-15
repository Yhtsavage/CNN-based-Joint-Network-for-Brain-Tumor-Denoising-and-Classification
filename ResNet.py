import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

class transpose(nn.Module):
    def __init__(self, in_channles, out_channles, kernel, padding, stride):
        super(transpose, self).__init__()
        self.tans1 = nn.ConvTranspose2d(in_channels=in_channles, out_channels=out_channles, kernel_size=kernel,
                                        padding=padding, stride = stride)

    def forward(self,X):
        X = self.tans1(X)
        return X

class basic_block(nn.Module):
    """基本残差块,由两层卷积构成"""
    def __init__(self,in_channles, num_channles, kernel_size=3, stride=1):
        """

        :param in_channles: 输入通道
        :param num_channles:  输出通道
        :param kernel_size: 卷积核大小
        :param stride: 卷积步长
        """
        super(basic_block, self).__init__()
        self.conv1=nn.Conv2d(in_channles, num_channles, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn1=nn.BatchNorm2d(num_channles)
        self.relu=nn.ReLU()
        self.conv2=nn.Conv2d(num_channles, num_channles, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn2=nn.BatchNorm2d(num_channles)
        if stride!=1 or in_channles!=num_channles:
            self.downsample=nn.Sequential(nn.Conv2d(in_channles, num_channles, kernel_size=1, stride=stride)
                                          , nn.BatchNorm2d(num_channles))
        else:
            self.downsample=nn.Sequential()
    def forward(self,inx):
        x=self.relu(self.bn1(self.conv1(inx)))
        x=self.bn2(self.conv2(x))
        out=x+self.downsample(inx)
        return F.relu(out)


class Resnet(nn.Module):
    def __init__(self,basicBlock,blockNums,nb_classes):
        super(Resnet, self).__init__()
        self.in_planes=64
        #输入层
        self.conv1=nn.Conv2d(3,self.in_planes,kernel_size=(7,7),stride=(2,2),padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(self.in_planes)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1=self._make_layers(basicBlock,blockNums[0],64,1)
        self.layer2=self._make_layers(basicBlock,blockNums[1],128,2)
        self.layer3=self._make_layers(basicBlock,blockNums[2],256,2)
        self.layer4=self._make_layers(basicBlock,blockNums[3],512,2)
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc=nn.Linear(512,nb_classes)
        self.de1 = nn.Sequential(transpose(512, 256, kernel=5, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.de2 = nn.Sequential(transpose(256, 128, kernel=4, stride=2, padding=2), nn.BatchNorm2d(128),
                            nn.ReLU())  # 变28 需要
        self.de3 = transpose(128, 64, kernel=3, stride=2, padding=1)  # 变55
        self.de4 = nn.Sequential(transpose(64, 64, kernel=4, stride=2, padding=0), nn.BatchNorm2d(64), nn.ReLU())  # 变112 需要
        self.de5 = transpose(64, 32, kernel=3, stride=2, padding=1)  # for resnet_18
        self.de6 = nn.Sequential(transpose(32, 3, kernel=2, stride=1, padding=0), nn.Sigmoid())
    def _make_layers(self,basicBlock,blockNum,plane,stride):
        """
        :param basicBlock: 基本残差块类
        :param blockNum: 当前层包含基本残差块的数目,resnet18每层均为2
        :param plane: 输出通道数
        :param stride: 卷积步长
        :return:
        """
        layers=[]
        for i in range(blockNum):
            if i==0:                    #每个残差层有两块 第一个块需要下采样，且需要加通道，步长为2 减小图片大小
                layer=basicBlock(self.in_planes,plane,3,stride=stride)
            else:
                layer=basicBlock(plane,plane,3,stride=1)
            layers.append(layer)
        self.in_planes=plane
        return nn.Sequential(*layers)

    def forward(self,inx):
        x0 = self.relu(self.bn1(self.conv1(inx)))#112
        x1 = self.maxpool(x0)#56
        x2 = self.layer1(x1)#56
        x3 =self.layer2(x2)#28
        x4 = self.layer3(x3)#14
        x_feature = self.layer4(x4)#7 #把全连接层去掉了 encoder要给decoder输入
        x = self.avgpool(x_feature)
        out = x.reshape(x.shape[0], -1)
        out = self.fc(out)
        de1 = self.de1(x_feature)#7到15
        de2 = self.de2(de1)# 15到28
        de3 = self.de3(de2+x3)# 28的相加
        de4 = self.de4(de3)# 55到112
        de5 = self.de5(de4+x0)# 112的相加
        de6 = self.de6(de5)
        return out, de6
resnet18=Resnet(basic_block,[2,2,2,2],2)

"""print(resnet18)
inx=torch.randn(32,1,256,256)
print(inx.shape)
cla,feature=resnet18(inx)
print(feature.shape)"""

"""from torchvision import models
resnet18_1 = models.resnet18(pretrained=False)
#resnet18_1.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
resnet18_1.fc = nn.Linear(512, 2, bias=True)
resnet18_feature_extractor = nn.Sequential(*list(resnet18_1.children())[:-2])
"""#print(resnet18_1)
"""print(resnet18_feature_extractor)
print(resnet18_1)
x = torch.rand((1,1,256,256))
y = resnet18_feature_extractor(x)#torch.Size([1, 512, 8, 8])
print(y.shape)"""
if __name__ == '__main__':
    print(resnet18.named_parameters())
    x = torch.rand((1, 3, 224, 224))
    y = resnet18(x)

    print(y[0].shape)
    print(y[1].shape)
    weights_dict = torch.load('./weights/model-28.pth')
    load_weights_dict = {k: v for k, v in weights_dict.items() if 'classif' in k}  # 读取模型的值
                         #if resnet18.state_dict()[k].numel() == v.numel()}  # 需要的权重值数量相同的话即可"""
    #resnet18.load_state_dict(load_weights_dict)
    #print(len(resnet18_1.named_parameters()))







