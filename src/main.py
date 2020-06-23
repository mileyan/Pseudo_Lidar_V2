import os
import shutil
import time

import configargparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

import disp_models
import logger
import models
import utils_func
from dataloader import KITTILoader3D
from dataloader import KITTILoader_dataset3d
from dataloader import SceneFlowLoader
from dataloader import listflowfile

parser = configargparse.ArgParser(description='PSMNet')
parser.add('-c', '--config', required=True,
           is_config_file=True, help='config file')

parser.add_argument('--save_path', type=str, default='',
                    help='path to save the log, tensorbaord and checkpoint')
# network
parser.add_argument('--data_type', default='depth', choices=['disparity', 'depth'],
                    help='the network can predict either disparity or depth')
parser.add_argument('--arch', default='SDNet', choices=['SDNet', 'PSMNet'],
                    help='Model Name, default: SDNet.')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity, the range of the disparity cost volume: [0, maxdisp-1]')
parser.add_argument('--down', type=float, default=2,
                    help='reduce x times resolution when build the depth cost volume')
parser.add_argument('--maxdepth', type=int, default=80,
                    help='the range of the depth cost volume: [1, maxdepth]')
# dataset
parser.add_argument('--kitti2015', action='store_true',
                    help='If false, use 3d kitti dataset. If true, use kitti stereo 2015, default: False')
parser.add_argument('--dataset', default='kitti', choices=['sceneflow', 'kitti'],
                    help='train with sceneflow or kitti')
parser.add_argument('--datapath', default='',
                    help='root folder of the dataset')
parser.add_argument('--split_train', default='Kitti/object/train.txt',
                    help='data splitting file for training')
parser.add_argument('--split_val', default='Kitti/object/subval.txt',
                    help='data splitting file for validation')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of training epochs')
parser.add_argument('--btrain', type=int, default=12,
                    help='training batch size')
parser.add_argument('--bval', type=int, default=4,
                    help='validation batch size')
parser.add_argument('--workers', type=int, default=8,
                    help='number of dataset workers')
# learning rate
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--lr_stepsize', nargs='+', type=int, default=[200],
                    help='drop lr in each step')
parser.add_argument('--lr_gamma', default=0.1, type=float,
                    help='gamma of the learning rate scheduler')
# resume
parser.add_argument('--resume', default=None,
                    help='path to a checkpoint')
parser.add_argument('--pretrain', default=None,
                    help='path to pretrained model')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='start epoch')
# evaluate
parser.add_argument('--evaluate', action='store_true',
                    help='do evaluation')
parser.add_argument('--calib_value', type=float, default=1017,
                    help='manually define focal length. (sceneflow does not have configuration)')
parser.add_argument('--dynamic_bs', action='store_true',
                    help='If true, dynamically calculate baseline from calibration file. If false, use 0.54')
parser.add_argument('--eval_interval', type=int, default=50,
                    help='evaluate model every n epochs')
parser.add_argument('--checkpoint_interval', type=int, default=5,
                    help='save checkpoint every n epoch.')
parser.add_argument('--generate_depth_map', action='store_true',
                    help='if true, generate depth maps and save the in save_path/depth_maps/{data_tag}/')
parser.add_argument('--data_list', default=None,
                    help='generate depth maps for all the data in this list')
parser.add_argument('--data_tag', default=None,
                    help='the suffix of the depth maps folder')
args = parser.parse_args()
best_RMSE = 1e10


def main():
    global best_RMSE

    # set logger
    log = logger.setup_logger(os.path.join(args.save_path, 'training.log'))
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    # set tensorboard
    writer = SummaryWriter(args.save_path + '/tensorboardx')

    # Data Loader
    if args.generate_depth_map:
        TrainImgLoader = None
        import dataloader.KITTI_submission_loader  as KITTI_submission_loader
        TestImgLoader = torch.utils.data.DataLoader(
            KITTI_submission_loader.SubmiteDataset(args.datapath, args.data_list, args.dynamic_bs),
            batch_size=args.bval, shuffle=False, num_workers=args.workers, drop_last=False)
    elif args.dataset == 'kitti':
        train_data, val_data = KITTILoader3D.dataloader(args.datapath, args.split_train, args.split_val,
                                                        kitti2015=args.kitti2015)
        TrainImgLoader = torch.utils.data.DataLoader(
            KITTILoader_dataset3d.myImageFloder(train_data, True, kitti2015=args.kitti2015, dynamic_bs=args.dynamic_bs),
            batch_size=args.btrain, shuffle=True, num_workers=args.workers, drop_last=False, pin_memory=True)
        TestImgLoader = torch.utils.data.DataLoader(
            KITTILoader_dataset3d.myImageFloder(val_data, False, kitti2015=args.kitti2015, dynamic_bs=args.dynamic_bs),
            batch_size=args.bval, shuffle=False, num_workers=args.workers, drop_last=False, pin_memory=True)
    else:
        train_data, val_data = listflowfile.dataloader(args.datapath)
        TrainImgLoader = torch.utils.data.DataLoader(
            SceneFlowLoader.myImageFloder(train_data, True, calib=args.calib_value),
            batch_size=args.btrain, shuffle=True, num_workers=args.workers, drop_last=False)
        TestImgLoader = torch.utils.data.DataLoader(
            SceneFlowLoader.myImageFloder(val_data, False, calib=args.calib_value),
            batch_size=args.bval, shuffle=False, num_workers=args.workers, drop_last=False)

    # Load Model
    if args.data_type == 'disparity':
        model = disp_models.__dict__[args.arch](maxdisp=args.maxdisp)
    elif args.data_type == 'depth':
        model = models.__dict__[args.arch](maxdepth=args.maxdepth, maxdisp=args.maxdisp, down=args.down)
    else:
        log.info('Model is not implemented')
        assert False

    # Number of parameters
    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    model = nn.DataParallel(model).cuda()
    torch.backends.cudnn.benchmark = True

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = MultiStepLR(optimizer, milestones=args.lr_stepsize, gamma=args.lr_gamma)

    if args.pretrain:
        if os.path.isfile(args.pretrain):
            log.info("=> loading pretrain '{}'".format(args.pretrain))
            checkpoint = torch.load(args.pretrain)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            log.info('[Attention]: Can not find checkpoint {}'.format(args.pretrain))

    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_RMSE = checkpoint['best_RMSE']
            scheduler.load_state_dict(checkpoint['scheduler'])
            log.info("=> loaded checkpoint '{}' (epoch {})"
                     .format(args.resume, checkpoint['epoch']))
        else:
            log.info('[Attention]: Can not find checkpoint {}'.format(args.resume))

    if args.generate_depth_map:
        os.makedirs(args.save_path + '/depth_maps/' + args.data_tag, exist_ok=True)

        tqdm_eval_loader = tqdm(TestImgLoader, total=len(TestImgLoader))
        for batch_idx, (imgL_crop, imgR_crop, calib, H, W, filename) in enumerate(tqdm_eval_loader):
            pred_disp = inference(imgL_crop, imgR_crop, calib, model)
            for idx, name in enumerate(filename):
                np.save(args.save_path + '/depth_maps/' + args.data_tag + '/' + name, pred_disp[idx][-H[idx]:, :W[idx]])
        import sys
        sys.exit()

    # evaluation
    if args.evaluate:
        evaluate_metric = utils_func.Metric()
        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L, calib) in enumerate(TestImgLoader):
            start_time = time.time()
            test(imgL_crop, imgR_crop, disp_crop_L, calib, evaluate_metric, model)

            log.info(evaluate_metric.print(batch_idx, 'EVALUATE') + ' Time:{:.3f}'.format(time.time() - start_time))
        import sys
        sys.exit()

    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()

        ## training ##
        train_metric = utils_func.Metric()
        tqdm_train_loader = tqdm(TrainImgLoader, total=len(TrainImgLoader))
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L, calib) in enumerate(tqdm_train_loader):
            # start_time = time.time()
            train(imgL_crop, imgR_crop, disp_crop_L, calib, train_metric, optimizer, model)
            # log.info(train_metric.print(batch_idx, 'TRAIN') + ' Time:{:.3f}'.format(time.time() - start_time))
        log.info(train_metric.print(0, 'TRAIN Epoch' + str(epoch)))
        train_metric.tensorboard(writer, epoch, token='TRAIN')
        # lw.update(train_metric.get_info(), epoch, 'Train')

        ## testing ##
        is_best = False
        if (epoch % args.eval_interval) == 0:
            test_metric = utils_func.Metric()
            tqdm_test_loader = tqdm(TestImgLoader, total=len(TestImgLoader))
            for batch_idx, (imgL_crop, imgR_crop, disp_crop_L, calib) in enumerate(tqdm_test_loader):
                # start_time = time.time()
                test(imgL_crop, imgR_crop, disp_crop_L, calib, test_metric, model)
                # log.info(test_metric.print(batch_idx, 'TEST') + ' Time:{:.3f}'.format(time.time() - start_time))
            log.info(test_metric.print(0, 'TEST Epoch' + str(epoch)))
            test_metric.tensorboard(writer, epoch, token='TEST')

            # SAVE
            is_best = test_metric.RMSELIs.avg < best_RMSE
            best_RMSE = min(test_metric.RMSELIs.avg, best_RMSE)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_RMSE': best_RMSE,
            'scheduler': scheduler.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best, epoch, folder=args.save_path)
    # lw.done()


def save_checkpoint(state, is_best, epoch, filename='checkpoint.pth.tar', folder='result/default'):
    torch.save(state, folder + '/' + filename)
    if is_best:
        shutil.copyfile(folder + '/' + filename, folder + '/model_best.pth.tar')
    if args.checkpoint_interval > 0 and (epoch + 1) % args.checkpoint_interval == 0:
        shutil.copyfile(folder + '/' + filename, folder + '/checkpoint_{}.pth.tar'.format(epoch + 1))


def train(imgL, imgR, depth, calib, metric_log, optimizer, model):
    model.train()
    calib = calib.float()

    imgL, imgR, depth, calib = imgL.cuda(), imgR.cuda(), depth.cuda(), calib.cuda()

    # ---------
    mask = (depth >= 1) * (depth <= 80)
    mask.detach_()
    # ----

    optimizer.zero_grad()

    output1, output2, output3 = model(imgL, imgR, calib)
    output1 = torch.squeeze(output1, 1)
    output2 = torch.squeeze(output2, 1)
    output3 = torch.squeeze(output3, 1)
    if args.data_type == 'disparity':
        output1 = disp2depth(output1, calib)
        output2 = disp2depth(output2, calib)
        output3 = disp2depth(output3, calib)
    loss = 0.5 * F.smooth_l1_loss(output1[mask], depth[mask], size_average=True) + 0.7 * F.smooth_l1_loss(
        output2[mask], depth[mask], size_average=True) + F.smooth_l1_loss(output3[mask], depth[mask],
                                                                          size_average=True)

    metric_log.calculate(depth, output3, loss=loss.item())
    loss.backward()
    optimizer.step()


def inference(imgL, imgR, calib, model):
    model.eval()
    imgL, imgR, calib = imgL.cuda(), imgR.cuda(), calib.float().cuda()

    with torch.no_grad():
        output = model(imgL, imgR, calib)
    if args.data_type == 'disparity':
        output = disp2depth(output, calib)
    pred_disp = output.data.cpu().numpy()

    return pred_disp


def test(imgL, imgR, depth, calib, metric_log, model):
    model.eval()
    calib = calib.float()
    imgL, imgR, calib, depth = imgL.cuda(), imgR.cuda(), calib.cuda(), depth.cuda()

    mask = (depth >= 1) * (depth <= 80)
    mask.detach_()
    with torch.no_grad():
        output3 = model(imgL, imgR, calib)
        output3 = torch.squeeze(output3, 1)

        if args.data_type == 'disparity':
            output3 = disp2depth(output3, calib)
        loss = F.smooth_l1_loss(output3[mask], depth[mask], size_average=True)

        metric_log.calculate(depth, output3, loss=loss.item())

    torch.cuda.empty_cache()
    return


def disp2depth(disp, calib):
    depth = calib[:, None, None] / disp.clamp(min=1e-8)
    return depth


if __name__ == '__main__':
    main()
