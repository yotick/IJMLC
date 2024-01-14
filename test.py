import numpy as np
import torch
import os
from PIL import Image
from skimage.io import imsave
from utils_didfuse import Test_fusion

# =============================================================================
# Test Details
# =============================================================================
device='cuda'
vispath = './Datasets/Test_data_TNO/VIS/'
irpath = './Datasets/Test_data_TNO/IR/'

# =============================================================================
# Test
# =============================================================================
for root,dirs,files in os.walk(irpath):
    with torch.no_grad():
        for file in files:
            Test_IR = Image.open(irpath+'IR'+file[2:]) # infrared image
            Test_Vis = Image.open(vispath+'VIS'+file[2:]) # visible image
            img_re=Test_fusion(IR = Test_IR,VIS = Test_Vis)
            imsave('./Test_result/Result/'+file[2:-4]+'.png',img_re)
