'''
Correct predicted depthmaps with sparse LiDAR ground-truths
by Graph-based Depth Correction (GDC)

Author: Yurong You
Date: Feb 2020
'''

from pykdtree.kdtree import KDTree
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gmres, cg
from scipy.sparse import eye as seye
from scipy.sparse import csr_matrix
from scipy import sparse
import numpy as np
import time
import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

def filter_mask(pc_rect):
    """Return index of points that lies within the region defined below."""
    valid_inds = (pc_rect[:, 2] < 80) * \
                 (pc_rect[:, 2] > 1) * \
                 (pc_rect[:, 0] < 40) * \
                 (pc_rect[:, 0] >= -40) * \
                 (pc_rect[:, 1] < 2.5) * \
                 (pc_rect[:, 1] >= -1)
    return valid_inds


GRID_SIZE = 0.1
index_field_sample = np.full(
    (35, int(80 / 0.1), int(80 / 0.1)), -1, dtype=np.int32)

def subsample_mask_by_grid(pc_rect):
    N = pc_rect.shape[0]
    perm = np.random.permutation(pc_rect.shape[0])
    pc_rect = pc_rect[perm]

    range_filter = filter_mask(pc_rect)
    pc_rect = pc_rect[range_filter]

    pc_rect_quantized = np.floor(pc_rect[:, :3] / GRID_SIZE).astype(np.int32)
    pc_rect_quantized[:, 0] = pc_rect_quantized[:, 0] \
        + int(80 / GRID_SIZE / 2)
    pc_rect_quantized[:, 1] = pc_rect_quantized[:, 1] + int(1 / GRID_SIZE)

    index_field = index_field_sample.copy()

    index_field[pc_rect_quantized[:, 1],
                pc_rect_quantized[:, 2], pc_rect_quantized[:, 0]] = np.arange(pc_rect.shape[0])
    mask = np.zeros(perm.shape, dtype=np.bool)
    mask[perm[range_filter][index_field[index_field >= 0]]] = 1
    return mask


def filter_theta_mask(pc_rect, low, high):
    # though if we have to do this precisely, we should convert
    # point clouds to velodyne space, here we just use those in rect space,
    # since actually the velodyne and the cameras are very close to each other.

    x, y, z = pc_rect[:, 0], pc_rect[:, 1], pc_rect[:, 2]
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arcsin(y / d)
    return (theta >= low) * (theta < high)


def depth2ptc(depth, calib):
    """Convert a depth_map to a pointcloud."""
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth]).reshape((3, -1)).T
    return calib.project_image_to_rect(points)


def GDC(pred_depth, gt_depth, calib,
        k=10,
        W_tol=1e-5,
        recon_tol=1e-4,
        verbose=False,
        method='gmres',
        consider_range=(-0.1, 3.0),
        subsample=False,
        ):
    """
    Returns the depth map after Graph-based Depth Correction (GDC).

    Parameters:
        pred_depth - predicted depthmap
        gt_depth - lidar depthmap (-1 means no groundtruth)
        calib - calibration object
        k - k used in KNN
        W_tol - tolerance in solving reconstruction weights
        recon_tol - tolerance used in gmres / cg
        debug - if in debug mode (more info will show)
        verbose - if True, more info will show
        method - use cg or gmres to solve the second step
        consider_range - perform LLDC only on points whose pitch angles are
            within this range
        subsample - whether subsampling points by grids

    Returns:
        new_depth_map - A refined depthmap with the same size of pred_depth
    """

    if verbose:
        print("warpping up depth infos...")

    ptc = depth2ptc(pred_depth, calib)
    consider_PL = (filter_mask(ptc) * filter_theta_mask(
        ptc, low=np.radians(consider_range[0]),
        high=np.radians(consider_range[1]))).reshape(pred_depth.shape)
    if subsample:
        subsample_mask = subsample_mask_by_grid(
            ptc).reshape(pred_depth.shape)
        consider_PL = consider_PL * subsample_mask


    consider_L = filter_mask(depth2ptc(gt_depth, calib)
                             ).reshape(gt_depth.shape)
    gt_mask = consider_L * consider_PL

    # We don't drastically move points.
    # This avoids numerical issues in solving linear equations.
    gt_mask[gt_mask] *= (np.abs(pred_depth[gt_mask] - gt_depth[gt_mask]) < 2)

    # we only consider points within certain ranges
    pred_mask = np.logical_not(gt_mask) * consider_PL

    x_info = np.concatenate((pred_depth[pred_mask], pred_depth[gt_mask]))
    gt_info = gt_depth[gt_mask]
    N_PL = pred_mask.sum()   # number of pseudo_lidar points
    N_L = gt_mask.sum()      # number of lidar points (groundtruth)
    ptc = np.concatenate(
        (ptc[pred_mask.reshape(-1)], ptc[gt_mask.reshape(-1)]))
    if verbose:
        print("N_PL={} N_L={}".format(N_PL, N_L))
        print("building up KDtree...")

    tree = KDTree(ptc)
    neighbors = tree.query(ptc, k=k+1)[1][:, 1:]

    if verbose:
        print("sovling W...")

    As = np.zeros((N_PL + N_L, k+2, k+2))
    bs = np.zeros((N_PL + N_L, k+2))
    As[:, :k, :k] = np.eye(k) * (1 + W_tol)
    As[:, k+1, :k] = 1
    As[:, :k, k+1] = 1
    bs[:, k+1] = 1
    bs[:, k] = x_info
    As[:, k, :k] = x_info[neighbors]
    As[:, :k, k] = x_info[neighbors]

    W = np.linalg.solve(As, bs)[:, :k]

    if verbose:
        avg = 0
        for i in range(N_PL):
            avg += np.abs(W[i, :k].dot(x_info[neighbors[i]]) - x_info[i])
        print("average reconstruction diff: {:.3e}".format(avg / N_PL))
        print("building up sparse W...")

    # We devide the sparse W matrix into 4 parts:
    # [W_PLPL, W_LPL]
    # [W_PLL , W_LL ]
    idx_PLPL = neighbors[:N_PL] < N_PL
    indptr_PLPL = np.concatenate(([0], np.cumsum(idx_PLPL.sum(axis=1))))
    W_PLPL = csr_matrix((W[:N_PL][idx_PLPL], neighbors[:N_PL]
                         [idx_PLPL], indptr_PLPL), shape=(N_PL, N_PL))

    idx_LPL = neighbors[:N_PL] >= N_PL
    indptr_LPL = np.concatenate(([0], np.cumsum(idx_LPL.sum(axis=1))))
    W_LPL = csr_matrix((W[:N_PL][idx_LPL], neighbors[:N_PL]
                        [idx_LPL] - N_PL, indptr_LPL), shape=(N_PL, N_L))

    idx_PLL = neighbors[N_PL:] < N_PL
    indptr_PLL = np.concatenate(([0], np.cumsum(idx_PLL.sum(axis=1))))
    W_PLL = csr_matrix((W[N_PL:][idx_PLL], neighbors[N_PL:]
                        [idx_PLL], indptr_PLL), shape=(N_L, N_PL))

    idx_LL = neighbors[N_PL:] >= N_PL
    indptr_LL = np.concatenate(([0], np.cumsum(idx_LL.sum(axis=1))))
    W_LL = csr_matrix((W[N_PL:][idx_LL], neighbors[N_PL:]
                       [idx_LL] - N_PL, indptr_LL), shape=(N_L, N_L))

    if verbose:
        print("reconstructing depth...")

    A = sparse.vstack((seye(N_PL) - W_PLPL, W_PLL))
    b = np.concatenate((W_LPL.dot(gt_info), gt_info - W_LL.dot(gt_info)))

    ATA = LinearOperator((A.shape[1], A.shape[1]),
                         matvec=lambda x: A.T.dot(A.dot(x)))
    method = cg if method == 'cg' else gmres
    x_new, info = method(ATA, A.T.dot(
        b), x0=x_info[:N_PL], tol=recon_tol)
    if verbose:
        print(info)
        print('solve in error: {}'.format(np.linalg.norm(A.dot(x_new) - b)))

    if subsample:
        new_depth_map = np.full_like(pred_depth, -1)
        new_depth_map[subsample_mask] = pred_depth[subsample_mask]
    else:
        new_depth_map = pred_depth.copy()
    new_depth_map[pred_mask] = x_new
    new_depth_map[gt_depth > 0] = gt_depth[gt_depth > 0]

    return new_depth_map
