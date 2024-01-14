device = 'cuda'
import sys
def process_command_args(arguments):

    # specifying default parameters

    batch_size = 50
    train_size = 30000
    learning_rate = 5e-4
    num_train_iters = 20000

    w_content = 10
    w_color = 0.5
    w_texture = 1
    w_tv = 2000

    dped_dir = 'dped/'
    vgg_dir = 'vgg_pretrained/imagenet-vgg-verydeep-19.mat'
    eval_step = 1000

    phone = ""

    for args in arguments:

        if args.startswith("models"):
            phone = args.split("=")[1]

        if args.startswith("batch_size"):
            batch_size = int(args.split("=")[1])

        if args.startswith("train_size"):
            train_size = int(args.split("=")[1])

        if args.startswith("learning_rate"):
            learning_rate = float(args.split("=")[1])

        if args.startswith("num_train_iters"):
            num_train_iters = int(args.split("=")[1])

        # -----------------------------------

        if args.startswith("w_content"):
            w_content = float(args.split("=")[1])

        if args.startswith("w_color"):
            w_color = float(args.split("=")[1])

        if args.startswith("w_texture"):
            w_texture = float(args.split("=")[1])

        if args.startswith("w_tv"):
            w_tv = float(args.split("=")[1])

        # -----------------------------------

        if args.startswith("dped_dir"):
            dped_dir = args.split("=")[1]

        if args.startswith("vgg_dir"):
            vgg_dir = args.split("=")[1]

        if args.startswith("eval_step"):
            eval_step = int(args.split("=")[1])


    if phone == "":
        print("\nPlease specify the camera models by running the script with the following parameter:\n")
        print("python train_model.py models={iphone,blackberry,sony}\n")
        sys.exit()

    if phone not in ["iphone", "sony", "blackberry"]:
        print("\nPlease specify the correct camera models:\n")
        print("python train_model.py models={iphone,blackberry,sony}\n")
        sys.exit()

    print("\nThe following parameters will be applied for CNN training:\n")

    print("Phone models:", phone)
    print("Batch size:", batch_size)
    print("Learning rate:", learning_rate)
    print("Training iterations:", str(num_train_iters))
    print()
    print("Content loss:", w_content)
    print("Color loss:", w_color)
    print("Texture loss:", w_texture)
    print("Total variation loss:", str(w_tv))
    print()
    print("Path to DPED dataset:", dped_dir)
    print("Path to VGG-19 network:", vgg_dir)
    print("Evaluation step:", str(eval_step))
    print()
    return phone, batch_size, train_size, learning_rate, num_train_iters, \
            w_content, w_color, w_texture, w_tv,\
            dped_dir, vgg_dir, eval_step


def process_test_model_args(arguments):

    phone = ""
    dped_dir = 'dped/'
    test_subset = "small"
    iteration = "all"
    resolution = "orig"
    use_gpu = "true"

    for args in arguments:

        if args.startswith("models"):
            phone = args.split("=")[1]

        if args.startswith("dped_dir"):
            dped_dir = args.split("=")[1]

        if args.startswith("test_subset"):
            test_subset = args.split("=")[1]

        if args.startswith("iteration"):
            iteration = args.split("=")[1]

        if args.startswith("resolution"):
            resolution = args.split("=")[1]

        if args.startswith("use_gpu"):
            use_gpu = args.split("=")[1]

    if phone == "":
        print("\nPlease specify the models by running the script with the following parameter:\n")
        print("python test_model.py models={iphone,blackberry,sony,iphone_orig,blackberry_orig,sony_orig}\n")
        sys.exit()

    return phone, dped_dir, test_subset, iteration, resolution, use_gpu


def get_resolutions():

    # IMAGE_HEIGHT, IMAGE_WIDTH

    res_sizes = {}

    res_sizes["iphone"] = [1536, 2048]
    res_sizes["iphone_orig"] = [1536, 2048]
    res_sizes["blackberry"] = [1560, 2080]
    res_sizes["blackberry_orig"] = [1560, 2080]
    res_sizes["sony"] = [1944, 2592]
    res_sizes["sony_orig"] = [1944, 2592]
    res_sizes["high"] = [1260, 1680]
    res_sizes["medium"] = [1024, 1366]
    res_sizes["small"] = [768, 1024]
    res_sizes["tiny"] = [600, 800]

    return res_sizes


def get_specified_res(res_sizes, phone, resolution):

    if resolution == "orig":
        IMAGE_HEIGHT = res_sizes[phone][0]
        IMAGE_WIDTH = res_sizes[phone][1]
    else:
        IMAGE_HEIGHT = res_sizes[resolution][0]
        IMAGE_WIDTH = res_sizes[resolution][1]

    IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 3

    return IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SIZE


def extract_crop(image, resolution, phone, res_sizes):

    if resolution == "orig":
        return image

    else:

        x_up = int((res_sizes[phone][1] - res_sizes[resolution][1]) / 2)
        y_up = int((res_sizes[phone][0] - res_sizes[resolution][0]) / 2)

        x_down = x_up + res_sizes[resolution][1]
        y_down = y_up + res_sizes[resolution][0]

        return image[y_up : y_down, x_up : x_down, :]

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats as st
def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis=2)
    return out_filter

from torch.autograd import Variable
import torch.nn.functional as Fu
def sobel(im):
    im = torch.unsqueeze(im,0)
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')  #
    sobel_kernel_x = sobel_kernel_x.reshape((1, 1, 3, 3))
    weight_x = Variable(torch.from_numpy(sobel_kernel_x)).cuda()
    edge_detect_x = Fu.conv2d(Variable(im), weight_x)
    sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype='float32')  #
    sobel_kernel_y = sobel_kernel_y.reshape((1, 1, 3, 3))
    weight_y = Variable(torch.from_numpy(sobel_kernel_y)).cuda()
    edge_detect_y = Fu.conv2d(Variable(im), weight_y)
    edge_detect = edge_detect_x + edge_detect_y

    return edge_detect
class Blur(nn.Module):
    def __init__(self, nc):
        super(Blur, self).__init__()
        self.nc = nc
        kernel = gauss_kernel(kernlen=21, nsig=3, channels=self.nc)
        kernel = torch.from_numpy(kernel).permute(2, 3, 0, 1)
        self.weight = nn.Parameter(data=kernel, requires_grad=False).to(device)

    def forward(self, x):
        if x.size(1) != self.nc:
            raise RuntimeError(
                "The channel of input [%d] does not match the preset channel [%d]" % (x.size(1), self.nc))
        x = F.conv2d(x, self.weight, stride=1, padding=10, groups=self.nc)
        return x

def SM_map(img_gray):
    c = 0.00000000001
    a = img_gray - torch.mean(img_gray)
    b = (a + torch.abs(a)) / 2 + c
    b = (b - torch.min(b)) / (torch.max(b) - torch.min(b) + c)
    return b
def SM_map_VIS(img_gray):
    c = 0.00000000001
    # img_gray = torch.pow(img_gray,10)
    a = img_gray - torch.mean(img_gray)
    b = (a + torch.abs(a)) / 2 + c
    b = (b - torch.min(b)) / (torch.max(b) - torch.min(b) + c)
    return b


mse_loss = torch.nn.MSELoss()

class SM_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Fuse_image , IR_image,saliency_map  ):
        a = Fuse_image.shape
        my_pixel_loss = 0
        mse_loss = torch.nn.MSELoss()
        for i in range(a[0]):
            Fuse_image_temp = Fuse_image[i, :, :, :]
            IR_image_temp = IR_image[i, :, :, :]
            saliency_map_temp = saliency_map[i, :, :, :]
            Fuse_image_temp = torch.squeeze(Fuse_image_temp, 0)
            IR_image_temp = torch.squeeze(IR_image_temp, 0)
            saliency_map_temp = torch.squeeze(saliency_map_temp, 0)

            #IR_saliency_map = SM_map(IR_image_temp)
            #saliency_map = torch.ceil(saliency_map_temp)

            my_pixel_loss_temp_IR = mse_loss( saliency_map * IR_image_temp , saliency_map * Fuse_image_temp )

            my_pixel_loss_temp = my_pixel_loss_temp_IR

            my_pixel_loss = my_pixel_loss + my_pixel_loss_temp

        my_pixel_loss = my_pixel_loss / a[0]

        return my_pixel_loss
class SM_loss2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Fuse_image, IR_image):
        a = Fuse_image.shape
        my_pixel_loss = 0

        for i in range(a[0]):
            Fuse_image_temp = Fuse_image[i, :, :, :]
            IR_image_temp = IR_image[i, :, :, :]
            Fuse_image_temp = torch.squeeze(Fuse_image_temp, 0)
            IR_image_temp = torch.squeeze(IR_image_temp, 0)

            IR_saliency_map = SM_map_VIS(IR_image_temp)
            IR_saliency_map = torch.ceil(IR_saliency_map)
            my_pixel_loss_temp_IR = mse_loss(IR_image_temp * IR_saliency_map, Fuse_image_temp*IR_saliency_map)
            my_pixel_loss_temp = my_pixel_loss_temp_IR
            my_pixel_loss = my_pixel_loss + my_pixel_loss_temp

        my_pixel_loss = my_pixel_loss / a[0]

        return my_pixel_loss
def gradient(input):
    # 用nn.Conv2d定义卷积操作
    conv_op = nn.Conv2d(1, 1, 3, bias=False,padding=0)
    # 定义算子参数 [0.,1.,0.],[1.,-4.,1.],[0.,1.,0.] Laplacian 四邻域 八邻域
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
    # 将算子转换为适配卷积操作的卷积核
    kernel = kernel.reshape((1, 1, 3, 3))
    # 给卷积操作的卷积核赋值
    conv_op.weight.data = (torch.from_numpy(kernel)).to(device).type(torch.float32)
    # 对图像进行卷积操作
    edge_detect = conv_op(input)
    # print(edge_detect.shape)
    # imgs = edge_detect.cpu().detach().numpy()
    # print(img.shape)
    # import cv2
    # for i in range(64):
    #     # print(i,imgs.shape)
    #     img = imgs[i, :, :]
    #     img = img.squeeze()
    #     min = np.amin(img)
    #     max = np.amax(img)
    #     img = (img - min) / (max - min)
    #     img = img * 255
    #     # print(img.shape)
    #
    #     cv2.imwrite('gradient/gradinet' + str(i) + '.jpg', img)

    return edge_detect


