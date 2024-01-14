import torch
from net import DenseFuseNet
import torchvision.transforms as transforms
import numpy as np
_tensor = transforms.ToTensor()


device='cuda'

def output_img(x):
    return x.cpu().detach().numpy()[0,0,:,:]

transforms = transforms.Compose([
    #transforms.Resize((256,256)),
    transforms.Grayscale(1),
    transforms.ToTensor(),
])
def Test_fusion(IR, VIS):
    model = DenseFuseNet().to(device)
    model.load_state_dict(torch.load(
            "./Train_result/best.pkl"
            )['weight'])
    model.eval()
    img_test1 = _tensor(IR).unsqueeze(0).to(device)
    img_test2 = _tensor(VIS).unsqueeze(0).to(device)

    img_re= model(t =img_test1,rgb =img_test2)

    return output_img(img_re)
