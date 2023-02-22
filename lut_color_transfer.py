import argparse
import torch
import os
import numpy as np
import cv2
from PIL import Image

from 3DLUT.models_x import *
from 3DLUT import torchvision_x_functional as TF_x
import torchvision.transforms.functional as TF


parser = argparse.ArgumentParser()

parser.add_argument("--image_dir", type=str, default="demo_images", help="directory of image")
parser.add_argument("--image_name", type=str, default="3.jpg", help="name of image")
parser.add_argument("--input_color_space", type=str, default="sRGB", help="input color space: sRGB or XYZ")
parser.add_argument("--model_dir", type=str, default="pretrained_models", help="directory of pretrained models")
parser.add_argument("--output_dir", type=str, default="demo_results", help="directory to save results")
opt = parser.parse_args()
opt.model_dir = opt.model_dir + '/' + opt.input_color_space
opt.image_path = opt.image_dir + '/' + opt.input_color_space + '/' + opt.image_name
os.makedirs(opt.output_dir, exist_ok=True)

# use gpu when detect cuda
cuda = True if torch.cuda.is_available() else False
# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

criterion_pixelwise = torch.nn.MSELoss()
LUT0 = Generator3DLUT_identity()
classifier = Classifier()
trilinear_ = TrilinearInterpolation() 

if cuda:
    LUT0 = LUT0.cuda()
    classifier = classifier.cuda()
    criterion_pixelwise.cuda()

# ----------
#  test
# ----------
# read image and transform to tensor
img = Image.open(opt.image_path)
img = TF.to_tensor(img).type(Tensor)
img = img.unsqueeze(0)

LUT = LUT0.LUT

# generate image
_, result = trilinear_(LUT, img)

# save image
ndarr = result.squeeze().mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
im = Image.fromarray(ndarr)
im.save('%s/result.jpg' % opt.output_dir, quality=95)


