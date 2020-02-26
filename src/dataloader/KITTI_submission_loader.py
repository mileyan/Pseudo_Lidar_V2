import os
import torchvision.transforms as transforms
import torch.nn.functional as F
import PIL.Image as Image
import numpy as np


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


def dataloader(filepath, split):
    left_fold = 'image_2/'
    right_fold = 'image_3/'
    calib_fold = 'calib/'
    with open(split, 'r') as f:
        image = [x.strip() for x in f.readlines() if len(x.strip())>0]

    left_test = [filepath + left_fold + img + '.png' for img in image]
    right_test = [filepath + right_fold + img + '.png' for img in image]
    calib_test = [filepath + calib_fold + img + '.txt' for img in image]

    return left_test, right_test, calib_test

def dynamic_baseline(calib_info):
    P3 =np.reshape(calib_info['P3'], [3,4])
    P =np.reshape(calib_info['P2'], [3,4])
    baseline = P3[0,3]/(-P3[0,0]) - P[0,3]/(-P[0,0])
    return baseline

class SubmiteDataset(object):
    def __init__(self, filepath, split, dynamic_bs=False, kitti2015=False):
        self.dynamic_bs = dynamic_bs
        left_fold = 'image_2/'
        right_fold = 'image_3/'
        calib_fold = 'calib/'
        with open(split, 'r') as f:
            image = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        image = sorted(image)

        if kitti2015:
            self.left_test = [filepath + '/' + left_fold + img + '_10.png' for img in image]
            self.right_test = [filepath + '/' + right_fold + img + '_10.png' for img in image]
            self.calib_test = [filepath + '/' + calib_fold + img + '.txt' for img in image]
        else:
            self.left_test = [filepath + '/' + left_fold + img + '.png' for img in image]
            self.right_test = [filepath + '/' + right_fold + img + '.png' for img in image]
            self.calib_test = [filepath + '/' + calib_fold + img + '.txt' for img in image]

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


    def __getitem__(self, item):
        left_img = self.left_test[item]
        right_img = self.right_test[item]
        calib_info = read_calib_file(self.calib_test[item])
        if self.dynamic_bs:
            calib = np.reshape(calib_info['P2'], [3, 4])[0, 0] * dynamic_baseline(calib_info)
        else:
            calib = np.reshape(calib_info['P2'], [3, 4])[0, 0] * 0.54
        imgL = Image.open(left_img).convert('RGB')
        imgR = Image.open(right_img).convert('RGB')
        imgL = self.trans(imgL)[None, :, :, :]
        imgR = self.trans(imgR)[None, :, :, :]
        # pad to (384, 1248)
        B, C, H, W = imgL.shape
        top_pad = 384 - H
        right_pad = 1248 - W
        imgL = F.pad(imgL, (0, right_pad, top_pad, 0), "constant", 0)
        imgR = F.pad(imgR, (0, right_pad, top_pad, 0), "constant", 0)
        filename = self.left_test[item].split('/')[-1][:-4]

        return imgL[0].float(), imgR[0].float(), calib.item(), H, W, filename


    def __len__(self):
        return len(self.left_test)
