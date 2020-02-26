import argparse
import os

import numpy as np
import tqdm


def pto_rec_map(velo_points, H=64, W=512, D=800):
    # depth, width, height
    valid_inds = (velo_points[:, 0] < 80) & \
                 (velo_points[:, 0] >= 0) & \
                 (velo_points[:, 1] < 50) & \
                 (velo_points[:, 1] >= -50) & \
                 (velo_points[:, 2] < 1) & \
                 (velo_points[:, 2] >= -2.5)
    velo_points = velo_points[valid_inds]

    x, y, z, i = velo_points[:, 0], velo_points[:, 1], velo_points[:, 2], velo_points[:, 3]
    x_grid = (x * D / 80.).astype(int)
    x_grid[x_grid < 0] = 0
    x_grid[x_grid >= D] = D - 1

    y_grid = ((y + 50) * W / 100.).astype(int)
    y_grid[y_grid < 0] = 0
    y_grid[y_grid >= W] = W - 1

    z_grid = ((z + 2.5) * H / 3.5).astype(int)
    z_grid[z_grid < 0] = 0
    z_grid[z_grid >= H] = H - 1

    depth_map = - np.ones((D, W, H, 4))
    depth_map[x_grid, y_grid, z_grid, 0] = x
    depth_map[x_grid, y_grid, z_grid, 1] = y
    depth_map[x_grid, y_grid, z_grid, 2] = z
    depth_map[x_grid, y_grid, z_grid, 3] = i
    depth_map = depth_map.reshape((-1, 4))
    depth_map = depth_map[depth_map[:, 0] != -1.0]
    return depth_map


def pto_ang_map(velo_points, H=64, W=512, slice=1):
    """
    :param H: the row num of depth map, could be 64(default), 32, 16
    :param W: the col num of depth map
    :param slice: output every slice lines
    """

    dtheta = np.radians(0.4 * 64.0 / H)
    dphi = np.radians(90.0 / W)

    x, y, z, i = velo_points[:, 0], velo_points[:, 1], velo_points[:, 2], velo_points[:, 3]

    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r = np.sqrt(x ** 2 + y ** 2)
    d[d == 0] = 0.000001
    r[r == 0] = 0.000001
    phi = np.radians(45.) - np.arcsin(y / r)
    phi_ = (phi / dphi).astype(int)
    phi_[phi_ < 0] = 0
    phi_[phi_ >= W] = W - 1

    theta = np.radians(2.) - np.arcsin(z / d)
    theta_ = (theta / dtheta).astype(int)
    theta_[theta_ < 0] = 0
    theta_[theta_ >= H] = H - 1

    depth_map = - np.ones((H, W, 4))
    depth_map[theta_, phi_, 0] = x
    depth_map[theta_, phi_, 1] = y
    depth_map[theta_, phi_, 2] = z
    depth_map[theta_, phi_, 3] = i
    depth_map = depth_map[0::slice, :, :]
    depth_map = depth_map.reshape((-1, 4))
    depth_map = depth_map[depth_map[:, 0] != -1.0]
    return depth_map


def gen_sparse_points(pl_data_path, args):
    pc_velo = np.fromfile(pl_data_path, dtype=np.float32).reshape((-1, 4))

    # depth, width, height
    valid_inds = (pc_velo[:, 0] < 120) & \
                 (pc_velo[:, 0] >= 0) & \
                 (pc_velo[:, 1] < 50) & \
                 (pc_velo[:, 1] >= -50) & \
                 (pc_velo[:, 2] < 1.5) & \
                 (pc_velo[:, 2] >= -2.5)
    pc_velo = pc_velo[valid_inds]

    return pto_ang_map(pc_velo, H=args.H, W=args.W, slice=args.slice)


def gen_sparse_points_all(args):
    outputfolder = args.sparse_pl_path
    os.makedirs(outputfolder, exist_ok=True)
    data_idx_list = sorted([x.strip() for x in os.listdir(args.pl_path) if x[-3:] == 'bin'])

    for data_idx in tqdm.tqdm(data_idx_list):
        sparse_points = gen_sparse_points(os.path.join(args.pl_path, data_idx), args)
        sparse_points = sparse_points.astype(np.float32)
        sparse_points.tofile(f'{outputfolder}/{data_idx}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Generate sparse pseudo-LiDAR points")
    parser.add_argument('--pl_path', default='/scratch/datasets', help='pseudo-lidar path')
    parser.add_argument('--sparse_pl_path', default='/scratch/datasets', help='sparsed pseudo lidar path')
    parser.add_argument('--slice', default=1, type=int)
    parser.add_argument('--H', default=64, type=int)
    parser.add_argument('--W', default=512, type=int)
    parser.add_argument('--D', default=700, type=int)
    args = parser.parse_args()

    gen_sparse_points_all(args)
