import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18
from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_res_unet(n_input=1, n_output=2, size=256):
    # Create an instance of the resnet18 model
    resnet18_model = resnet18(pretrained=True)

    # Transfer the model to the appropriate device
    resnet18_model = resnet18_model.to(device)

    # Create the body using create_body from fastai
    body = create_body(resnet18_model, pretrained=True, n_in=n_input, cut=-2)

    # Create the DynamicUnet model
    net_G = DynamicUnet(body, n_output, (size, size)).to(device)

    return net_G