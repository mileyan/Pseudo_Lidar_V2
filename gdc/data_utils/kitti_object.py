''' Helper class and functions for loading KITTI objects

Author: Charles R. Qi
Date: September 2017

Modified by Yurong You
Date: June 2019
'''
import os
import data_utils.kitti_util as utils


class kitti_object(object):
    '''Load and parse object data into a usable format.'''

    def __init__(self, root_dir=None,
        lidar_dir='velodyne',
        label_dir='label_2', calib_dir='calib',
        image_dir='image_2'):

        self.image_dir = os.path.join(root_dir, image_dir) \
            if root_dir is not None else image_dir
        self.label_dir = os.path.join(root_dir, label_dir) \
            if root_dir is not None else label_dir
        self.calib_dir = os.path.join(root_dir, calib_dir) \
            if root_dir is not None else calib_dir
        self.lidar_dir = os.path.join(root_dir, lidar_dir) \
            if root_dir is not None else lidar_dir

    def get_image(self, idx):
        img_filename = os.path.join(self.image_dir, '%06d.png'%(idx))
        return utils.load_image(img_filename)

    def get_right_image(self, idx):
        img_filename = os.path.join(self.right_image_dir, '%06d.png'%(idx))
        return utils.load_image(img_filename)

    def get_lidar(self, idx):
        lidar_filename = os.path.join(self.lidar_dir, '%06d.bin'%(idx))
        return utils.load_velo_scan(lidar_filename)

    def get_calibration(self, idx):
        calib_filename = os.path.join(self.calib_dir, '%06d.txt'%(idx))
        return utils.Calibration(calib_filename)

    def get_label_objects(self, idx):
        label_filename = os.path.join(self.label_dir, '%06d.txt'%(idx))
        return utils.read_label(label_filename)

def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=2.0):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:,0]<xmax) & (pts_2d[:,0]>=xmin) & \
        (pts_2d[:,1]<ymax) & (pts_2d[:,1]>=ymin)
    fov_inds = fov_inds & (pc_velo[:,0]>clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds,:]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo

def get_rect_in_image_fov(pc_rect, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=2.0):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d = calib.project_rect_to_image(pc_rect)
    fov_inds = (pts_2d[:,0]<xmax) & (pts_2d[:,0]>=xmin) & \
        (pts_2d[:,1]<ymax) & (pts_2d[:,1]>=ymin)
    fov_inds = fov_inds & (pc_rect[:,2]>clip_distance)
    imgfov_pc = pc_rect[fov_inds,:]
    if return_more:
        return imgfov_pc, pts_2d, fov_inds
    else:
        return imgfov_pc

