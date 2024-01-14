import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import sobel
from torchvision.transforms import ToPILImage


class Gradient_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Fuse_image, IR_image):
        mse_loss = torch.nn.MSELoss()
        a = Fuse_image.shape
        Grad_loss = 0
        for i in range(a[0]):
            Fuse_image_temp = Fuse_image[i,:,:,:]
            IR_image_temp = IR_image[i,:,:,:]
            #Fuse_image_temp = torch.squeeze(Fuse_image_temp, 0)
            #Fuse_image_temp = torch.squeeze(Fuse_image_temp, 0)
            #IR_image_temp = torch.squeeze(IR_image_temp, 0)
            #IR_image_temp = torch.squeeze(IR_image_temp, 0)

            #Fuse_image_temp = Fuse_image_temp.cpu()
            #_imIRage_temp = IR_image_temp.cpu()
            #Fuse_image_temp = Fuse_image_temp.detach().numpy()
            #IR_image_temp = IR_image_temp.detach().numpy()
            Fuse_grad_map = sobel(Fuse_image_temp)
            IR_grad_map = sobel(IR_image_temp)
            Grad_loss_temp_gray = mse_loss(Fuse_grad_map,IR_grad_map)

            Grad_loss_temp = Grad_loss_temp_gray

            Grad_loss = Grad_loss + Grad_loss_temp

        Grad_loss = Grad_loss / a[0]

        return Grad_loss


