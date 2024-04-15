import torch
from torch import nn, optim
class PatchDiscriminator(nn.Module):
  def __init__(self,input,num_filters=64,depth=3):
    super().__init__()
    model = [self.get_layers(input,num_filters,norm = False)]
    model+= [self.get_layers(num_filters * 2 ** i,num_filters * 2 ** (i+1),s=1 if (i==depth-1) else 2) for i in range(depth)]
    model+= [self.get_layers(num_filters*2**depth,1,s=1,norm=False,act=False)]

    self.model = nn.Sequential(*model)

  def get_layers(self,n1,n2,k=4,s = 2,p=1, norm = True , act = True):
    layers = [nn.Conv2d(in_channels=n1 , out_channels = n2 ,kernel_size = k , stride = s , padding = p ,bias= not norm)]
    if norm: layers+=[nn.BatchNorm2d(num_features=n2)]
    if act: layers+=[nn.LeakyReLU(0.2,inplace=True)]
    return nn.Sequential(*layers)

  def forward(self,x):
    return self.model(x)
