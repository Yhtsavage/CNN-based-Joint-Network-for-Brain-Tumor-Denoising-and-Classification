from torch import nn
import torch
from ResNet import *
from ResNet_noConnection import *
#from TransposeDecoder import *
#from LeNet import *

class EncoderDecoeder(nn.Module):
    def __init__(self, encoder, decoder, classific, **kwargs):
        super(EncoderDecoeder, self).__init__()
        self.encoder = encoder
        self.classif = classific
        self.decoder = decoder
    def forward(self, X):
        tumor, denoise = self.classif(X)
        tumor = F.softmax(tumor,dim=1)
        return denoise, tumor

Classifier = resnet18_noConnection
Encoder = None
Decoder = None

net = EncoderDecoeder(Encoder, Decoder, Classifier)

if __name__ == '__main__':
    from tensorboardX import SummaryWriter

    writer = SummaryWriter(log_dir="runs/TB")
    x = torch.rand((1,3,224,224))
    model = net
    writer.add_graph(net, input_to_model=x)
    writer.close()



