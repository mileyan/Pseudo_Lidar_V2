import os
import argparse
import os.path as osp
import numpy as np
from data_utils.kitti_util import Calibration
from tqdm.auto import tqdm
from multiprocessing import Process, Queue, Pool

def filter_height(ptc_velo, threshold=1):
    return ptc_velo[ptc_velo[:, 2] < threshold]

def depth2ptc(depth, calib):
    vu = np.indices(depth.shape).reshape(2, -1).T
    vu[:, 0], vu[:, 1] = vu[:, 1], vu[:, 0].copy()
    uv = vu
    uv_depth = np.column_stack((uv.astype(float), depth.reshape(-1)))
    return calib.project_image_to_rect(uv_depth)

parser = argparse.ArgumentParser(description='gen pointcloud from depthmap')
parser.add_argument('--output_path', type=str)
parser.add_argument('--input_path', type=str)
parser.add_argument('--calib_path', type=str,
                    help='path to calibration files')
parser.add_argument('--split_file', type=str,
                    help='indices of scene to be corrected')
parser.add_argument('--i', type=int, default=None)
parser.add_argument('--threads', type=int, default=4)
args = parser.parse_args()

def convert_and_save(args, i):
    depth_map = np.load(osp.join(
                args.input_path, "{:06d}.npy".format(i)))
    calib = Calibration(
        osp.join(args.calib_path, "{:06d}.txt".format(i)))
    ptc = filter_height(
        calib.project_rect_to_velo(depth2ptc(depth_map, calib)))
    ptc = np.hstack((ptc, np.ones((ptc.shape[0], 1)))).astype(np.float32)
    with open(osp.join(args.output_path, '{:06d}.bin'.format(i)), 'wb') as f:
        ptc.tofile(f)

if __name__ == "__main__":
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if args.i is not None:
        convert_and_save(args, args.i)
        # i = args.i
        # depth_map = np.load(osp.join(args.input_path, "{:06d}.npy".format(i)))
        # calib = Calibration(osp.join(args.calib_path, "{:06d}.txt".format(i)))
        # ptc = filter_height(calib.project_rect_to_velo(
        #     depth2ptc(depth_map, calib)))
        # # do remember to use float32!!!，
        # ptc = np.hstack((ptc, np.ones((ptc.shape[0], 1)))).astype(np.float32)
        # with open(osp.join(args.output_path, '{:06d}.bin'.format(i)), 'wb') as f:
        #         ptc.tofile(f)
    else:
        with open(args.split_file) as f:
            idx_list = [int(x.strip())
                for x in f.readlines() if len(x.strip()) > 0]
        pbar = tqdm(total=len(idx_list))

        def update(*a):
            pbar.update()
        pool = Pool(args.threads)
        res = []
        for i in idx_list:
            res.append((i, pool.apply_async(convert_and_save, args=(args, i),
                                            callback=update)))

        pool.close()
        pool.join()
        pbar.clear(nolock=False)
        pbar.close()
