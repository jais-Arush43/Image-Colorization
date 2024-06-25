import torch
from torch import nn, optim
from torchvision import transforms
from Generator import Generator
from Discriminator import PatchDiscriminator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GANLoss(nn.Module):
  def __init__(self,gan_mode='vanilla',real_label=1.0,fake_label=0.0):
    super().__init__()
    self.register_buffer('real_label',torch.tensor(real_label))
    self.register_buffer('fake_label',torch.tensor(fake_label))
    if gan_mode == 'vanilla':
      self.loss = nn.BCEWithLogitsLoss()
    elif gan_mode == 'lsgan':
      self.loss = nn.MSELoss()


  def get_labels(self,preds,target_is_real):
    if target_is_real:
      labels = self.real_label
    else:
      labels = self.fake_label

    return labels.expand_as(preds)

  def __call__(self,preds,target_is_real):
    labels = self.get_labels(preds,target_is_real)
    loss = self.loss(preds,labels)
    return loss

def init_weights(net,init='norm',gain=0.02):

  def init_function(layer):
    classname = layer.__class__.__name__
    if hasattr(layer,'weight') and 'Conv' in classname:
      if init == 'norm':
        nn.init.normal_(layer.weight.data, mean=0.0 , std=gain)
      elif init == 'xavier':
        nn.init.xavier_normal_(layer.weight.data, gain=gain)
      elif init == 'kaiming':
        nn.init.kaiming_normal_(layer.weight.data,a=0,mode='fan_in')

      if hasattr(layer,'bias') and layer.bias is not None:
        nn.init.constant_(layer.bias.data,0.0)

    elif 'BatchNorm2d' in classname:
      nn.init.normal_(layer.weight.data,mean=1.0,std=gain)
      nn.init.constant_(layer.bias.data,0.0)

  net.apply(init_function)
  print(f"model is initialized with {init} initialization")
  return net

def init_model(model,device):
  model = model.to(device)
  model = init_weights(model)
  return model


class ColorizationModel(nn.Module):
  def __init__(self,net_G= None,lr_G=2e-4,lr_D=2e-4,
               beta1=0.9,beta2=0.999,lambda_L1=100.):
    super().__init__()
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.lambda_L1 = lambda_L1
    # Initialization
    if net_G is None:
      self.net_G = init_model(Generator(input_c=1,output_c=2,n_down=8,num_filters=64),self.device)
    else:
      self.net_G = net_G.to(self.device)
    self.net_D = init_model(PatchDiscriminator(input=3,depth=3,num_filters=64),self.device)

    # Loss Function Call
    self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
    self.L1criterion = nn.L1Loss()

    # Optimizers
    self.opt_G = optim.Adam(self.net_G.parameters(),lr=lr_G,betas=(beta1,beta2))
    self.opt_D = optim.Adam(self.net_D.parameters(),lr=lr_D,betas=(beta1,beta2))
    # Gradients
  def set_requires_grad(self,model,requires_grad=True):
     for p in model.parameters():
       p.requires_grad = requires_grad
    # Setting Input into model
  def setup_input(self,data):
      self.L = data['L'].to(self.device)
      self.ab = data['ab'].to(self.device)
    # Outputs of Generator
  def forward(self,input_tensor):
      self.fake_color = self.net_G(input_tensor)
    # Loss Function of Discriminator
  def backward_D(self):
      fake_image = torch.cat([self.L,self.fake_color],dim=1)
      fake_preds = self.net_D(fake_image.detach())
      self.loss_D_fake = self.GANcriterion(fake_preds,False)
      real_image = torch.cat([self.L,self.ab],dim=1)
      real_preds = self.net_D(real_image)
      self.loss_D_real = self.GANcriterion(real_preds,True)
      self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
      self.loss_D.backward()
    # Loss Function of Generator
  def backward_G(self):
      fake_image = torch.cat([self.L,self.fake_color],dim=1)
      fake_preds = self.net_D(fake_image)
      self.loss_G_GAN = self.GANcriterion(fake_preds,True)
      self.loss_G_L1 = self.L1criterion(self.fake_color,self.ab) * self.lambda_L1
      self.loss_G = self.loss_G_GAN + self.loss_G_L1
      self.loss_G.backward()

  def optimize(self):
      self.forward(self.L)
      self.net_D.train()
      self.set_requires_grad(self.net_D,True)
      self.opt_D.zero_grad()
      self.backward_D()
      self.opt_D.step()

      self.net_G.train()
      self.set_requires_grad(self.net_D,False)
      self.opt_G.zero_grad()
      self.backward_G()
      self.opt_G.step()

