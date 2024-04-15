import numpy as np
import glob
import time
from PIL import Image
from pathlib import Path
from skimage.color import rgb2lab,lab2rgb
import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from fastai.data.external import untar_data,URLs
coco_path = untar_data(URLs.COCO_SAMPLE)
coco_path = str(coco_path) + "/train_sample"

path = coco_path

paths = glob.glob(path + "/*.jpg")
np.random.seed(1234)
paths_subset = np.random.choice(paths  , 10_000,  replace=False)
rand_idxs = np.random.permutation(10_000)
train_idxs = rand_idxs[:8000]
val_idxs = rand_idxs[8000:]
train_paths = paths_subset[train_idxs]
val_paths = paths_subset[val_idxs]
print(len(train_paths), len(val_paths))

SIZE=256
class ColorizationDataset(Dataset):
  def __init__(self,paths,split='train'):
    if split =='train':
      self.transforms=transforms.Compose([transforms.Resize((SIZE,SIZE),Image.BICUBIC),transforms.RandomHorizontalFlip()])
    elif split =='val':
      self.transforms=transforms.Resize((SIZE,SIZE),Image.BICUBIC)

    self.split =split
    self.size = SIZE
    self.paths =paths

  def __getitem__(self,idx):
    img = Image.open(self.paths[idx]).convert("RGB")
    img = self.transforms(img)
    img = np.array(img)
    img_lab = rgb2lab(img).astype("float32")
    img_lab = transforms.ToTensor()(img_lab)
    L = img_lab[[0], ...] / 50. - 1.
    ab = img_lab[[1, 2], ...] / 110.
    return {'L':L ,'ab':ab}

  def __len__(self):
    return len(self.paths)

def make_dataloaders(batch_size = 16 ,n_workers=4,pin_memory=True, **kwargs):
      dataset = ColorizationDataset(**kwargs)
      dataloader = DataLoader(dataset,batch_size=batch_size,num_workers=n_workers,pin_memory=pin_memory)
      return dataloader
  
train_set = make_dataloaders(paths=train_paths,split='train')
val_set = make_dataloaders(paths=val_paths,split='val')
data = next(iter(train_set))
Ls, abs_ = data['L'], data['ab']
print(Ls.shape, abs_.shape)
print(len(train_set), len(val_set))