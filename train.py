import torchvision
from torchvision import transforms
import torch
from torch import nn
import torch.optim as optim
import os
from Nbranch.ssim import SSIM
from Nbranch.loss_network import LossNetwork
import time
from tqdm import tqdm
from net import DenseFuseNet
import kornia
def SM_map(img_gray):
    c = 0.00000000001
    a = img_gray - (torch.mean(img_gray))
    b = (a + torch.abs(a)) / 2 + c
    b = (b - torch.min(b)) / (torch.max(b) - torch.min(b) + c)
    return b
device = "cuda"


L1Loss = nn.L1Loss()
ssim = SSIM()


with torch.no_grad():
    loss_network = LossNetwork()
    loss_network.to(device)
loss_network.eval()
# =============================================================================
# Hyperparameters Setting
# =============================================================================
train_data_path = 'E:\dataset256/'
# train_data_path = '.\\Datasets\\Train_data_FLIR\\'

root_VIS = train_data_path + 'VIS\\'
root_IR = train_data_path + 'IR\\'
# root_VISmask = train_data_path + 'VISmask\\'

train_path = '.\\Train_result\\'

batch_size = 7
epochs = 5
lr = 0.0005
Train_Image_Number = len(os.listdir(train_data_path + 'IR/IR'))

Iter_per_epoch = (Train_Image_Number % batch_size != 0) + Train_Image_Number // batch_size
# =============================================================================
# Preprocessing and dataset establishment
# =============================================================================

transforms = transforms.Compose([
    #transforms.Resize((256,256)),
    transforms.Grayscale(1),
    transforms.ToTensor(),
])

torch.cuda.manual_seed(42)

Data_VIS = torchvision.datasets.ImageFolder(root_VIS, transform=transforms)
dataloader_VIS = torch.utils.data.DataLoader(Data_VIS, batch_size, shuffle=False)

Data_IR = torchvision.datasets.ImageFolder(root_IR, transform=transforms)
dataloader_IR = torch.utils.data.DataLoader(Data_IR, batch_size, shuffle=False)

# =============================================================================
# Models
# =============================================================================
model = DenseFuseNet()
is_cuda = True
if is_cuda:
    model = model.cuda()

print(model)

optimizer = optim.Adam(model.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [epochs // 3, epochs // 3 * 2], gamma=0.1)

# =============================================================================
# Training
# =============================================================================
print('============ Training Begins ===============')
img_num = (len(Data_IR))
print("Load ---- {} pairs of images: KAIST:[{}]".format(img_num, len(Data_IR)))


min_loss = 100
s_time = time.time()
steps = len(dataloader_IR)
k_num = len(dataloader_IR) * 2


loss = torch.zeros(1)

for iteration in range(epochs):

    data_iter_VIS = iter(dataloader_VIS)
    data_iter_IR = iter(dataloader_IR)

    tqdms = tqdm(range(int(steps)))

    for step in tqdms:

        if step < k_num:
            data_VIS, _ = next(data_iter_VIS)
            data_IR, _ = next(data_iter_IR)

        data_VIS = data_VIS.cuda()
        data_IR = data_IR.cuda()

        optimizer.zero_grad()
        # =====================================================================
        # Calculate loss
        # =====================================================================
        img_re = model(rgb=data_VIS, t=data_IR)
        map = SM_map(data_IR)
        loss1 = L1Loss(data_IR * map, img_re * map)
        loss1 = loss1 * 50
        loss2 = L1Loss(
                kornia.filters.SpatialGradient()(img_re * (1 - map)),
                kornia.filters.SpatialGradient()(data_VIS * (1 - map))
                )
        loss2 = loss2 * 100
        loss3 = L1Loss(img_re * (1 - map), data_VIS * (1 - map))
        loss3 = loss3 * 10
        mse_loss_VF = 1 - 1*ssim(data_IR, img_re)
        mse_loss_IF = 1 - 1*ssim(data_VIS, img_re)
        ssim_loss = mse_loss_IF + mse_loss_VF

        loss = ssim_loss + loss1 + loss2 + loss3

        loss.backward()
        optimizer.step()

        los = loss.item()

        e_time = time.time() - s_time
        last_time = epochs * int(steps) * (e_time / (iteration * int(steps) + step + 1)) - e_time

        tqdms.set_description('%d MSGP[%.5f %.5f %.5f %.5f %.5f] T[%d:%d:%d] lr:%.4f ' %
                              (iteration, loss.item(),loss1.item(), loss2.item(),loss3.item(), ssim_loss.item(),
                               last_time / 3600, last_time / 60 % 60, last_time % 60,
                               optimizer.param_groups[0]['lr'] * 1000))
    # Save Weights and result
    if min_loss > loss.item():
        min_loss = loss.item()
        torch.save({'weight': model.state_dict(), 'epoch': iteration, 'batch_index': step},
                   os.path.join(train_path, 'best.pkl'))
        print('[%d] - Best models is saved -' % (iteration))
    #
    #torch.save({'weight': model.state_dict(), 'epoch': iteration, 'batch_index': step},
    #           os.path.join(train_path, str(iteration) + '.pkl'))

    if (iteration + 1) % 10 == 0 and iteration != 0:
        torch.save({'weight': model.state_dict(), 'epoch': iteration, 'batch_index': step},
                   os.path.join(train_path, 'model_weight_new.pkl'))
        print('[%d] - models is saved -' % (iteration))



