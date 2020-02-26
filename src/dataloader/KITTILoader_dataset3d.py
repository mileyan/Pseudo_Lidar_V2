import random

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    return np.load(path).astype(np.float32)


def read_calib_file(filepath):
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

def kitti2015_disparity_loader(filepath, calib):
    disp = np.array(Image.open(filepath))/256.
    depth = np.zeros_like(disp)
    mask = disp > 0
    depth[mask] = calib / disp[mask]
    return depth


def dynamic_baseline(calib_info):
    P3 =np.reshape(calib_info['P3'], [3,4])
    P =np.reshape(calib_info['P2'], [3,4])
    baseline = P3[0,3]/(-P3[0,0]) - P[0,3]/(-P[0,0])
    return baseline

class myImageFloder(data.Dataset):
    def __init__(self, data, training, kitti2015=False, dynamic_bs=False, loader=default_loader, dploader=disparity_loader):
        left, right, left_depth, left_calib = data
        self.left = left
        self.dynamic_bs = dynamic_bs
        self.right = right
        self.depth = left_depth
        self.calib = left_calib
        self.loader = loader
        self.kitti2015 = kitti2015
        self.dploader = dploader
        self.training = training
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        depth = self.depth[index]
        calib_info = read_calib_file(self.calib[index])
        if self.dynamic_bs:
            calib = np.reshape(calib_info['P2'], [3, 4])[0, 0] * dynamic_baseline(calib_info)
        else:
            calib = np.reshape(calib_info['P2'], [3, 4])[0, 0] * 0.54

        left_img = self.loader(left)
        right_img = self.loader(right)
        if self.kitti2015:
            dataL = kitti2015_disparity_loader(depth, calib)
        else:
            dataL = self.dploader(depth)

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

            # left_img = left_img.crop((w - 1232, h - 368, w, h))
            # right_img = right_img.crop((w - 1232, h - 368, w, h))
            left_img = left_img.crop((w - 1200, h - 352, w, h))
            right_img = right_img.crop((w - 1200, h - 352, w, h))
            w1, h1 = left_img.size

            # dataL1 = dataL[h - 368:h, w - 1232:w]
            dataL = dataL[h - 352:h, w - 1200:w]

            left_img = self.transform(left_img)
            right_img = self.transform(right_img)

        dataL = torch.from_numpy(dataL).float()
        return left_img.float(), right_img.float(), dataL.float(), calib.item()

    def __len__(self):
        return len(self.left)
