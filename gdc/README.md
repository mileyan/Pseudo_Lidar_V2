# dependency
* numpy
* scipy
* opencv-python
* tqdm
* scikit-image
* pykdtree

# Usage

## Get groundtruth depthmap (skip this step if the depthmaps are provided)
```
python ptc2depthmap.py --output_path <output_path> \
    --input_path <input_pointcloud_path> \
    --calib_path <KITTI_calib_folder> \
    --image_path <KITTI_image_folder> \
    --split_file --split_file <split_file>  --threads <thread_number>
```

## Run batch GDC on predicted depth maps
```
python main_batch.py --input_path <path_to_pred_depth_map> \
    --calib_path <KITTI_calib_folder> \
    --gt_depthmap_path <path_to_gt_depth_map>\
    --output_path <output_path> --threads <thread_number> \
    --split_file <split_file> (<otherargs>)
```

## Get pointclouds from corrected depth map

```
python depthmap2ptc.py --output_path <output_path> \
    --input_path <input_depthmap_path> \
    --calib_path <KITTI_calib_folder> \
    --split_file <split_file> --threads <thread_number>
```

The split file contains ids of scenes on which we are to run GDC, e.g.
```
000000
000003
000007
000009
000010
000011
...
```

## Sparcify point clouds
Since PointRCNN model exploits the sparcity of the point-clouds, when we apply it on pseudo-LiDAR point-clouds (which are dense), we follow the procedure below to slice the point-clouds to make it have similar sparse property of LiDAR. We found using the sparsed pseudo-LiDAR point-clouds improves the 3D detection performance of PointRCNN model.

### Sparse pseudo-LiDAR point-clouds to 64 lines
```
python sparsify.py --calib_path <KITTI_calib_folder> \
    --image_path <KITTI_image_folder> --ptc_path <pointcloud_folder> \
    --split_file <split_file> --W 1024 --slice 1 --H 64 --threads <thread_number>
```

### Simulate 4-beam LiDAR
To extract 4-line LiDAR from the velodyne data provided by KITTI, run
```
python sparsify.py --calib_path <KITTI_calib_folder> \
    --image_path <KITTI_image_folder> --ptc_path <pointcloud_folder> \
    --W 1024 --H 64 --line_spec 5 7 9 11 \
    --store_line_map_dir <temporary_store_path>
```
Note that this sparcify method is slightly diffrent from `sparsify.py` in the src folder, in that we store the 4 beam LiDAR angular map and put it back after GDC. The `<temporary_store_path>` is used to store the 4-beam ground-truth in angular map.

### Sparsify the corrected point clouds
```
python sparsify.py --calib_path <KITTI_calib_folder> \
    --image_path <KITTI_image_folder> --ptc_path <corrected_pointcloud_folder>\
    --W 1024 --slice 1 --H 64 --fill_in_map_dir <temporary_store_path>\
    --fill_in_spec 5 7 9 11
```
