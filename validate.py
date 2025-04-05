import os
import time
import random
import numpy as np
import logging
import argparse
import shutil
import zlib
import glob

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter

from util import config, transform
from util.common_util import AverageMeter, intersectionAndUnionGPU, find_free_port
from util.data_util import collate_fn_limit, collation_fn_voxelmean, collation_fn_voxelmean_tta
from util.logger import get_logger
from util.lr import MultiStepWithWarmup, PolyLR, PolyLRwithWarmup, Constant

from util.nuscenes import nuScenes
from util.semantic_kitti import SemanticKITTI
from util.dataset import RobotCar
from util.waymo import Waymo

from functools import partial
import pickle
import yaml
from torch_scatter import scatter_mean
import spconv.pytorch as spconv

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation Inference')
    parser.add_argument('--config', type=str, default='config/semantic_kitti/semantic_kitti_unet32_spherical_transformer.yaml', help='config file')
    parser.add_argument('--weight', type=str, default='/home/ljc/SphereFormer/runs/model_semantic_kitti.pth', help='path to pretrained model')
    parser.add_argument('--input_path', type=str, default='/home/ljc/Dataset/KITTI', help='path to input point cloud')
    parser.add_argument('--output_path', type=str, default='output.txt', help='path to output file')
    parser.add_argument('opts', help='see config/s3dis/s3dis_stratified_transformer.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg

def inference(model, dataset, args):

    model.eval() # important: set the model to eval mode
    torch.cuda.empty_cache()

    with torch.no_grad():
        for i in range(len(dataset)):
            # Load the data
            coord, xyz, feat, target, offset, reconstruct_idx, file_name = dataset[i]

            #Move to GPU
            coord = coord.cuda()
            xyz = xyz.cuda()
            feat = feat.cuda()
            target = target.cuda()
            batch = torch.zeros(coord.shape[0]).long().cuda() # single batch
            spatial_shape = np.clip((coord.max(0)[0] + 1).cpu().numpy(), 128, None)
            sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, 1)
            output = model(sinput, xyz, batch)
            output = output[reconstruct_idx, :]

            class_confidence, output = output.max(1)
            predicted_labels = output.cpu().numpy()
            xyz = xyz[reconstruct_idx, :].cpu().numpy() # get original cloud points using reconstruct_idx
            target = dataset[i][3][reconstruct_idx].cpu().numpy()

            # Save results to the output file
            with open(args.output_path, 'w') as f:
                for i in range(len(xyz)):
                    f.write(f'{xyz[i,0]:.6f} {xyz[i,1]:.6f} {xyz[i,2]:.6f} {target[i].item()} {predicted_labels[i]}\n')

            print(f"Results saved to {args.output_path}")

def main():
    args = get_parser()

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True

    # get model
    if args.arch == 'unet_spherical_transformer':
        from model.unet_spherical_transformer import Semantic as Model

        args.patch_size = np.array([args.voxel_size[i] * args.patch_size for i in range(3)]).astype(np.float32)
        window_size = args.patch_size * args.window_size
        window_size_sphere = np.array(args.window_size_sphere)
        model = Model(input_c=args.input_c,
            m=args.m,
            classes=args.classes,
            block_reps=args.block_reps,
            block_residual=args.block_residual,
            layers=args.layers,
            window_size=window_size,
            window_size_sphere=window_size_sphere,
            quant_size=window_size / args.quant_size_scale,
            quant_size_sphere=window_size_sphere / args.quant_size_scale,
            rel_query=args.rel_query,
            rel_key=args.rel_key,
            rel_value=args.rel_value,
            drop_path_rate=args.drop_path_rate,
            window_size_scale=args.window_size_scale,
            grad_checkpoint_layers=args.grad_checkpoint_layers,
            sphere_layers=args.sphere_layers,
            a=args.a,
        )
    else:
        raise Exception('architecture {} not supported yet'.format(args.arch))

    model = torch.nn.DataParallel(model.cuda()) # Remove distributed training

    # Load pretrained model
    if args.weight:
        if os.path.isfile(args.weight):
            print("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            print("=> loaded weight '{}'".format(args.weight))
        else:
            print("=> no weight found at '{}'".format(args.weight))
            return

    # Create Dataset object for single scan
    if args.data_name == 'nuscenes':
        dataset = RobotCar(args.data_root,
            voxel_size=args.voxel_size,
            split='test', # Ensure no augmentation
            return_ref=True,
            rotate_aug=False,
            flip_aug=False,
            scale_aug=False,
            transform_aug=False,
            trans_std=[0, 0, 0],
            ignore_label=args.ignore_label,
            voxel_max=args.voxel_max,
            xyz_norm=args.xyz_norm,
            pc_range=args.get("pc_range", None),
            use_tta=False,
            vote_num=1,
            single_file=args.input_path # Force to load specific file
        )

    elif args.data_name == 'semantic_kitti':
        dataset = RobotCar(args.data_root,
            voxel_size=args.voxel_size,
            split='test', # Ensure no augmentation
            return_ref=True,
            rotate_aug=False,
            flip_aug=False,
            scale_aug=False,
            transform_aug=False,
            trans_std=[0, 0, 0],
            elastic_aug=False,
            elastic_params=[[0.12, 0.4], [0.8, 3.2]],
            ignore_label=args.ignore_label,
            voxel_max=args.voxel_max,
            xyz_norm=args.xyz_norm,
            pc_range=args.get("pc_range", None),
            use_tta=False,
            vote_num=1,
            # single_file=args.input_path # Force to load specific file
        )

    elif args.data_name == 'waymo':
        dataset = Waymo(args.data_root,
            voxel_size=args.voxel_size,
            split='val', # Ensure no augmentation
            return_ref=True,
            rotate_aug=False,
            flip_aug=False,
            scale_aug=False,
            transform_aug=False,
            trans_std=[0, 0, 0],
            elastic_aug=False,
            elastic_params=[[0.12, 0.4], [0.8, 3.2]],
            ignore_label=args.ignore_label,
            voxel_max=args.voxel_max,
            xyz_norm=args.xyz_norm,
            pc_range=args.get("pc_range", None),
            use_tta=False,
            vote_num=1,
            single_file=args.input_path # Force to load specific file
        )

    else:
        raise ValueError("The dataset {} is not supported.".format(args.data_name))

    inference(model, dataset, args)


if __name__ == '__main__':
    import gc
    gc.collect()
    main()
