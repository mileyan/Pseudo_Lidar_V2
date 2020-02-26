IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(filepath, train_file, val_file, kitti2015=False):
    left_fold = 'image_2/'
    right_fold = 'image_3/'
    if kitti2015:
        depth_L = 'disp_occ_0/'
    else:
        depth_L = 'depth_map/'
    calib = 'calib/'

    with open(train_file, 'r') as f:
        train_idx = [x.strip() for x in f.readlines() if len(x.strip())>0]
    with open(val_file, 'r') as f:
        val_idx = [x.strip() for x in f.readlines() if len(x.strip())>0]
    if kitti2015:
        left_train = [filepath + '/' + left_fold + img + '_10.png' for img in train_idx]
        right_train = [filepath + '/' + right_fold + img + '_10.png' for img in train_idx]
        depth_train = [filepath + '/' + depth_L + img + '_10.png' for img in train_idx]
        calib_train = [filepath + '/' + calib + img + '.txt' for img in train_idx]

        left_val = [filepath + '/' + left_fold + img + '_10.png' for img in val_idx]
        right_val = [filepath + '/' + right_fold + img + '_10.png' for img in val_idx]
        depth_val = [filepath + '/' + depth_L + img + '_10.png' for img in val_idx]
        calib_val = [filepath + '/' + calib + img + '.txt' for img in val_idx]
    else:
        left_train = [filepath + '/' + left_fold + img + '.png' for img in train_idx]
        right_train = [filepath + '/' + right_fold + img + '.png' for img in train_idx]
        depth_train = [filepath + '/' + depth_L + img + '.npy' for img in train_idx]
        calib_train = [filepath + '/' + calib + img + '.txt' for img in train_idx]

        left_val = [filepath + '/' + left_fold + img + '.png' for img in val_idx]
        right_val = [filepath + '/' + right_fold + img + '.png' for img in val_idx]
        depth_val = [filepath + '/' + depth_L + img + '.npy' for img in val_idx]
        calib_val = [filepath + '/' + calib + img + '.txt' for img in val_idx]

    return [left_train, right_train, depth_train, calib_train], [left_val, right_val, depth_val, calib_val]
