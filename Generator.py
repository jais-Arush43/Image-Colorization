import torch
from torch import nn
class UnetBlock(nn.Module):
  def __init__(self,n1,n2, submodule=None ,input=None, outermost=False , innermost=False ,dropout=False):
    super().__init__()
    self.outermost=outermost
    if input is None : input = n1
    downconv = nn.Conv2d(in_channels= input ,out_channels= n2, kernel_size=4 ,stride=2 , padding=1 , bias=False)
    downrelu = nn.LeakyReLU(0.2,inplace=True)
    downnorm = nn.BatchNorm2d(num_features = n2)
    uprelu = nn.ReLU(inplace=True)
    upnorm = nn.BatchNorm2d(num_features = n1)

    # Now let's create the foundational architecture

    if innermost:
      upconv = nn.ConvTranspose2d(in_channels = n2 , out_channels = n1 , kernel_size = 4, stride = 2, padding = 1 ,bias = False)
      downward = [downrelu , downconv]
      upward = [uprelu,upconv,upnorm]
      model =  downward + upward

    elif outermost:
      upconv = nn.ConvTranspose2d(in_channels = n2 * 2 , out_channels = n1 , kernel_size = 4, stride = 2, padding = 1)
      downward = [downconv]
      upward = [uprelu,upconv,nn.Tanh()]
      model = downward + [submodule] + upward

    else:
       upconv = nn.ConvTranspose2d(in_channels = n2 * 2 , out_channels = n1 , kernel_size = 4, stride = 2, padding = 1 ,bias = False)
       downward = [downrelu,downconv,downnorm]
       upward = [uprelu,upconv,upnorm]
       if dropout: upward +=[nn.Dropout(0.5)]
       model = downward + [submodule] + upward

    self.model=nn.Sequential(*model)

  def forward(self,x):
    if self.outermost:
      return self.model(x)
    else:
      return torch.cat([x,self.model(x)],1)
  
class Generator(nn.Module):
  def __init__(self,input_c=1,output_c=2, n_down=8, num_filters=64):
    super().__init__()
    unet_block = UnetBlock(num_filters * 8 , num_filters * 8 , innermost=True)
    for _ in range(n_down-5):
      unet_block = UnetBlock(num_filters * 8 , num_filters * 8 ,submodule=unet_block , dropout=True)

    new_filters = num_filters * 8
    for _ in range(3):
      unet_block = UnetBlock(new_filters // 2, new_filters , submodule=unet_block)
      new_filters //= 2

    self.model = UnetBlock(output_c,new_filters,outermost=True,input=input_c,submodule=unet_block)

  def forward(self,x):
    return self.model(x)
