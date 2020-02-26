import sys
sys.path.insert(0, '..')

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

from LLDC_dev import LLDC_amgx as LLDC
from data_utils.kitti_util import Calibration
import os
import os.path as osp
import time
import numpy as np
from tqdm.auto import tqdm


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load(idx, pred_path='/home/yy785/datasets/sdn_fix_train',
         gt_path='/scratch/datasets/KITTI/object/training/depth_map_L4',
         calib_path='/scratch/datasets/KITTI/object/training/calib/'):
    pred = np.load(osp.join(pred_path, "{:06d}.npy".format(idx)))
    gt = np.load(osp.join(gt_path, "{:06d}.npy".format(idx)))
    calib = Calibration(osp.join(calib_path, "{:06d}.txt".format(idx)))
    return pred, gt, calib

if __name__ == "__main__":
    np.random.seed(9696)
    avg_time = 0
    count = 1
    sample = np.random.permutation(
        [int(f) for f in open('../image_sets/val.txt').readlines()])[:count]
    # sample = [int(f) for f in open('../image_sets/val.txt').readlines()]

    avg_wrap_up_time = AverageMeter()
    avg_tree_time = AverageMeter()
    avg_step1_time = AverageMeter()
    avg_build_sparse_W_time = AverageMeter()
    avg_step2_time = AverageMeter()
    avg_total_time = AverageMeter()
    avg_d2p_PL_time = AverageMeter()
    avg_subsample_time = AverageMeter()

    # warmup GPU
    pred, gt, calib = load(0)
    # start = time.time()
    _, (wrap_up_time, d2p_PL_time,
        subsample_time, tree_time, step1_time,
        build_sparse_W_time, step2_time, total_time) = \
        LLDC(pred, gt, calib,
             k=10,
             W_tol=1e-4,
             recon_tol=5e-4,
             debug=False,
             verbose=False,
             method='cg',
             consider_range=(-0.1, 3),
             subsample=True,
             collect_time_info=True)

    for inx in tqdm(sample):
        pred, gt, calib = load(inx)
        # start = time.time()
        _, (wrap_up_time, d2p_PL_time,
            subsample_time, tree_time, step1_time,
            build_sparse_W_time, step2_time, total_time) = \
                LLDC(pred, gt, calib,
                    k=10,
                    W_tol=1e-4,
                    recon_tol=5e-4,
                    debug=False,
                    verbose=False,
                    method='cg',
                    consider_range=(-0.1, 3),
                    subsample=True,
                    collect_time_info=True)
        # print(f"wrap_up_time:        {wrap_up_time * 1000:7.3f} ms")
        # print(f"d2p_PL_time:         {d2p_PL_time * 1000:7.3f} ms")
        # print(f"subsample_time:      {subsample_time * 1000:7.3f} ms")
        # print(f"tree_time:           {tree_time * 1000:7.3f} ms")
        # print(f"step1_time:          {step1_time * 1000:7.3f} ms")
        # print(f"build_sparse_W_time: {build_sparse_W_time * 1000:7.3f} ms")
        # print(f"step2_time:          {step2_time * 1000:7.3f} ms")
        # print(f"total_time:          {total_time * 1000:7.3f} ms")
        avg_wrap_up_time.update(wrap_up_time)
        avg_d2p_PL_time.update(d2p_PL_time)
        avg_subsample_time.update(subsample_time)
        avg_tree_time.update(tree_time)
        avg_step1_time.update(step1_time)
        avg_build_sparse_W_time.update(build_sparse_W_time)
        avg_step2_time.update(step2_time)
        avg_total_time.update(total_time)


    print(f"avg_wrap_up_time:        {avg_wrap_up_time.avg * 1000:7.3f} ms")
    print(f"avg_d2p_PL_time:         {avg_d2p_PL_time.avg * 1000:7.3f} ms")
    print(f"avg_subsample_time:      {avg_subsample_time.avg * 1000:7.3f} ms")
    print(f"avg_tree_time:           {avg_tree_time.avg * 1000:7.3f} ms")
    print(f"avg_step1_time:          {avg_step1_time.avg * 1000:7.3f} ms")
    print(f"avg_build_sparse_W_time: {avg_build_sparse_W_time.avg * 1000:7.3f} ms")
    print(f"avg_step2_time:          {avg_step2_time.avg * 1000:7.3f} ms")
    print(f"avg_total_time:          {avg_total_time.avg * 1000:7.3f} ms")
