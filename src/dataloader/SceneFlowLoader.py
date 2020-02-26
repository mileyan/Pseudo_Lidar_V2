import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
from . import readpfm as rp
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def disparity_loader(path):
    return rp.readPFM(path)


class myImageFloder(data.Dataset):
    def __init__(self, data, training, calib=1417., loader=default_loader, dploader=disparity_loader):
        left, right, left_disparity, = data
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        self.calib=calib

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL, scaleL = self.dploader(disp_L)
        dataL = np.ascontiguousarray(dataL, dtype=np.float32)
        if '15mm' in left:
            calib = self.calib/2.33*0.54
        else:
            calib = self.calib * 0.54

        mask = dataL >= 1.
        dataL[1-mask] = 0
        dataL[mask] = calib / np.clip(dataL[mask], 1., None)
        dataL = torch.from_numpy(dataL)

        if self.training:
            w, h = left_img.size
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            left_img = self.transform(left_img)
            right_img = self.transform(right_img)

        else:
            w, h = left_img.size
            left_img = left_img.crop((w - 960, h - 544, w, h))
            right_img = right_img.crop((w - 960, h - 544, w, h))
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
            dataL_pad = torch.zeros((dataL.shape[0]+4,dataL.shape[1]))
            dataL_pad[4:,:] = dataL
            dataL = dataL_pad

        return left_img.float(), right_img.float(), dataL.float(), float(calib)

    def __len__(self):
        return len(self.left)
