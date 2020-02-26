import random

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


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
    disp = np.array(Image.open(filepath)) / 256.
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
    def __init__(self, filepath, split, knn_root, filter_high=-1, kitti2015=False, dynamic_bs=False, loader=default_loader,
                 dploader=disparity_loader):
        self.filter_high = filter_high
        self.dynamic_bs = dynamic_bs
        with open(split, 'r') as f:
            self.split = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
            self.split = sorted(self.split)

        self.left_fold = 'image_2/'
        self.right_fold = 'image_3/'
        if kitti2015:
            self.depth_fold = 'disp_occ_0/'
        else:
            self.depth_fold = 'depth_map/'
        self.calib_fold = 'calib/'
        self.root = filepath
        self.knn_root = knn_root

        self.loader = loader
        self.kitti2015 = kitti2015
        self.dploader = dploader
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    def __getitem__(self, index):
        left = self.root + '/' + self.left_fold + '/' + self.split[index] + '.png'
        right = self.root + '/' + self.right_fold + '/' + self.split[index] + '.png'
        calib = self.root + '/' + self.calib_fold + '/' + self.split[index] + '.txt'
        depth_knn = self.knn_root + '/' + self.split[index] + '.npy'
        depth_gt = self.root + '/' + self.depth_fold + '/' + self.split[index] + '.npy'

        calib_info = read_calib_file(calib)
        if self.dynamic_bs:
            calib = np.reshape(calib_info['P2'], [3, 4])[0, 0] * dynamic_baseline(calib_info)
        else:
            calib = np.reshape(calib_info['P2'], [3, 4])[0, 0] * 0.54

        left_img = self.loader(left)
        right_img = self.loader(right)
        if self.kitti2015:
            dataL_gt = kitti2015_disparity_loader(depth_gt, calib)
        else:
            dataL_gt = self.dploader(depth_gt)
        dataL_knn = self.dploader(depth_knn)
        if self.filter_high > 0:
            dataL_knn[:self.filter_high, :] = -1

        w, h = left_img.size
        th, tw = 256, 512

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
        right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

        dataL_gt = dataL_gt[y1:y1 + th, x1:x1 + tw]
        dataL_knn = dataL_knn[y1:y1 + th, x1:x1 + tw]

        left_img = self.transform(left_img)
        right_img = self.transform(right_img)

        dataL_gt = torch.from_numpy(dataL_gt).float()
        dataL_knn = torch.from_numpy(dataL_knn).float()
        return left_img.float(), right_img.float(), dataL_gt.float(), dataL_knn, calib.item()

    def __len__(self):
        return len(self.split)



class myImageFloderVal(data.Dataset):
    def __init__(self, filepath, split, kitti2015=False, dynamic_bs=False, loader=default_loader,
                 dploader=disparity_loader):
        self.dynamic_bs = dynamic_bs
        with open(split, 'r') as f:
            self.split = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
            self.split = sorted(self.split)

        self.left_fold = 'image_2/'
        self.right_fold = 'image_3/'
        if kitti2015:
            self.depth_fold = 'disp_occ_0/'
        else:
            self.depth_fold = 'depth_map/'
        self.calib_fold = 'calib/'
        self.root = filepath

        self.loader = loader
        self.kitti2015 = kitti2015
        self.dploader = dploader
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    def __getitem__(self, index):
        left = self.root + '/' + self.left_fold + '/' + self.split[index] + '.png'
        right = self.root + '/' + self.right_fold + '/' + self.split[index] + '.png'
        calib = self.root + '/' + self.calib_fold + '/' + self.split[index] + '.txt'
        depth = self.root + '/' + self.depth_fold + '/' + self.split[index] + '.npy'

        calib_info = read_calib_file(calib)
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

        w, h = left_img.size

        left_img = left_img.crop((w - 1200, h - 352, w, h))
        right_img = right_img.crop((w - 1200, h - 352, w, h))

        dataL = dataL[h - 352:h, w - 1200:w]

        left_img = self.transform(left_img)
        right_img = self.transform(right_img)

        dataL = torch.from_numpy(dataL).float()
        return left_img.float(), right_img.float(), dataL.float(), calib.item()

    def __len__(self):
        return len(self.split)
