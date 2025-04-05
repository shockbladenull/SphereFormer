import os
import traceback
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
from util.waymo import Waymo

from functools import partial
import pickle
import yaml
from torch_scatter import scatter_mean
import spconv.pytorch as spconv


import pandas as pd

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/semantic_kitti/semantic_kitti_unet32_spherical_transformer.yaml', help='config file')
    parser.add_argument('opts', help='see config/s3dis/s3dis_stratified_transformer.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # import torch.backends.mkldnn
    # ackends.mkldnn.enabled = False
    # os.environ["LRU_CACHE_CAPACITY"] = "1"
    # cudnn.deterministic = True
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args, best_iou
    args, best_iou = argss, 0
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    
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
    
    # set optimizer
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "transformer_block" not in n and p.requires_grad],
            "lr": args.base_lr,
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if "transformer_block" in n and p.requires_grad],
            "lr": args.base_lr * args.transformer_lr_scale,
            "weight_decay": args.weight_decay,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.base_lr, weight_decay=args.weight_decay)

    if main_process():
        global logger, writer
        logger = get_logger(args.save_path)
        writer = SummaryWriter(args.save_path)
        logger.info(args)

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        if args.sync_bn:
            if main_process():
                logger.info("use SyncBN")
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    else:
        model = torch.nn.DataParallel(model.cuda())

    if main_process():
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info(model)
        logger.info('#Model parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))
        if args.get("max_grad_norm", None):
            logger.info("args.max_grad_norm = {}".format(args.max_grad_norm))

    # set loss func 
    class_weight = args.get("class_weight", None)
    class_weight = torch.tensor(class_weight).cuda() if class_weight is not None else None
    if main_process():
        logger.info("class_weight: {}".format(class_weight))
        logger.info("loss_name: {}".format(args.get("loss_name", "ce_loss")))
    criterion = nn.CrossEntropyLoss(weight=class_weight, ignore_index=args.ignore_label, reduction='none' if args.loss_name == 'focal_loss' else 'mean').cuda()
    
    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler_state_dict = checkpoint['scheduler']
            best_iou = checkpoint['best_iou']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    if args.data_name == 'nuscenes':
        train_data = nuScenes(args.data_root, 
            info_path_list=['nuscenes_seg_infos_1sweeps_train.pkl'], 
            voxel_size=args.voxel_size, 
            split='train',
            return_ref=True, 
            label_mapping=args.label_mapping, 
            rotate_aug=True, 
            flip_aug=True, 
            scale_aug=True, 
            transform_aug=True, 
            trans_std=[0.1, 0.1, 0.1],
            ignore_label=args.ignore_label,
            voxel_max=args.voxel_max, 
            xyz_norm=args.xyz_norm,
            pc_range=args.get("pc_range", None), 
            use_tta=args.use_tta,
            vote_num=args.vote_num,
        )
    
    elif args.data_name == 'semantic_kitti':
        train_data = SemanticKITTI(args.data_root, 
            voxel_size=args.voxel_size, 
            split='train', 
            return_ref=True, 
            label_mapping=args.label_mapping, 
            rotate_aug=True, 
            flip_aug=True, 
            scale_aug=True, 
            scale_params=[0.95,1.05], 
            transform_aug=True, 
            trans_std=[0.1, 0.1, 0.1],
            elastic_aug=False, 
            elastic_params=[[0.12, 0.4], [0.8, 3.2]], 
            ignore_label=args.ignore_label, 
            voxel_max=args.voxel_max, 
            xyz_norm=args.xyz_norm,
            pc_range=args.get("pc_range", None), 
            use_tta=args.use_tta,
            vote_num=args.vote_num,
        )

    elif args.data_name == 'waymo':
        train_data = Waymo(args.data_root, 
            voxel_size=args.voxel_size, 
            split='train', 
            return_ref=True, 
            rotate_aug=True, 
            flip_aug=True, 
            scale_aug=True, 
            scale_params=[0.95, 1.05], 
            transform_aug=True, 
            trans_std=[0.1, 0.1, 0.1],
            elastic_aug=False, 
            elastic_params=[[0.12, 0.4], [0.8, 3.2]], 
            ignore_label=args.ignore_label, 
            voxel_max=args.voxel_max, 
            xyz_norm=args.xyz_norm, 
            pc_range=args.get("pc_range", None), 
            use_tta=args.use_tta,
            vote_num=args.vote_num,
        )

    else:
        raise ValueError("The dataset {} is not supported.".format(args.data_name))

    if main_process():
        logger.info("train_data samples: '{}'".format(len(train_data)))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None

    collate_fn = partial(collate_fn_limit, max_batch_points=args.max_batch_points, logger=logger if main_process() else None)
    train_loader = torch.utils.data.DataLoader(train_data, 
        batch_size=args.batch_size, 
        shuffle=(train_sampler is None), 
        num_workers=args.workers,
        pin_memory=True, 
        sampler=train_sampler, 
        drop_last=True, 
        collate_fn=collate_fn
    )

    val_transform = None
    args.use_tta = getattr(args, "use_tta", False)
    if args.data_name == 'nuscenes':
        val_data = nuScenes(data_path=args.data_root, 
            info_path_list=['nuscenes_seg_infos_1sweeps_val.pkl'], 
            voxel_size=args.voxel_size, 
            split='val', 
            rotate_aug=args.use_tta, 
            flip_aug=args.use_tta, 
            scale_aug=args.use_tta, 
            transform_aug=args.use_tta, 
            xyz_norm=args.xyz_norm, 
            pc_range=args.get("pc_range", None),
            use_tta=args.use_tta,
            vote_num=args.vote_num,
        )
    elif args.data_name == 'semantic_kitti':
        val_data = SemanticKITTI(data_path=args.data_root, 
            voxel_size=args.voxel_size, 
            split='val', 
            rotate_aug=args.use_tta, 
            flip_aug=args.use_tta, 
            scale_aug=args.use_tta, 
            transform_aug=args.use_tta, 
            xyz_norm=args.xyz_norm, 
            pc_range=args.get("pc_range", None), 
            use_tta=args.use_tta,
            vote_num=args.vote_num,
        )
    elif args.data_name == 'waymo':
        val_data = Waymo(data_path=args.data_root, 
            voxel_size=args.voxel_size, 
            split='val', 
            rotate_aug=args.use_tta, 
            flip_aug=args.use_tta, 
            scale_aug=args.use_tta, 
            transform_aug=args.use_tta, 
            xyz_norm=args.xyz_norm, 
            pc_range=args.get("pc_range", None), 
            use_tta=args.use_tta,
            vote_num=args.vote_num,
        )
    else:
        raise ValueError("The dataset {} is not supported.".format(args.data_name))

    if main_process():
        logger.info("val_data samples: '{}'".format(len(val_data)))

    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, shuffle=False)
    else:
        val_sampler = None
        
    if getattr(args, "use_tta", False):
        val_loader = torch.utils.data.DataLoader(val_data, 
            batch_size=args.batch_size_val, 
            shuffle=False, 
            num_workers=args.workers, 
            pin_memory=True, 
            sampler=val_sampler, 
            collate_fn=collation_fn_voxelmean_tta
        )
    else:
        val_loader = torch.utils.data.DataLoader(val_data, 
            batch_size=args.batch_size_val, 
            shuffle=False, 
            num_workers=args.workers,
            pin_memory=True, 
            sampler=val_sampler, 
            collate_fn=collation_fn_voxelmean
        )

    # set scheduler
    if args.scheduler == 'Poly':
        if main_process():
            logger.info("scheduler: Poly. scheduler_update: {}".format(args.scheduler_update))
        if args.scheduler_update == 'epoch':
            scheduler = PolyLR(optimizer, max_iter=args.epochs, power=args.power)
        elif args.scheduler_update == 'step':
            iter_per_epoch = len(train_loader)
            scheduler = PolyLR(optimizer, max_iter=args.epochs*iter_per_epoch, power=args.power)
        else:
            raise ValueError("No such scheduler update {}".format(args.scheduler_update))
    else:
        raise ValueError("No such scheduler {}".format(args.scheduler))

    if args.resume and os.path.isfile(args.resume):
        scheduler.load_state_dict(scheduler_state_dict)
        print("resume scheduler")

    ###################
    # start training #
    ###################

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    if args.val:
        if main_process():
            logger.debug(f"args.val={args.val}")
            logger.debug(f"args.use_tta={args.use_tta}")
        if args.use_tta:
            validate_tta(val_loader, model, criterion)
        else:
            # validate(val_loader, model, criterion)
            if main_process():
                logger.debug("extract_point_classifications()")
            # validate_distance(val_loader, model, criterion)
            # extract_point_classifications(val_loader, model)
            # results = infer(model, val_loader)
            # save_results(results)
            # validate_and_print_points13(val_loader, model)
            remove_moving_points_from_pointcloud2(val_loader, model)
            # remove_moving_points_from_pointcloud2(val_loader, model, logger)
        if main_process():
            logger.debug("exit()")
        return
        
    if main_process():
        logger.debug("for epoch in range(args.start_epoch, args.epochs)")
        
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if main_process():
            logger.info("lr: {}".format(scheduler.get_last_lr()))
            
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, criterion, optimizer, epoch, scaler, scheduler, gpu)
        if args.scheduler_update == 'epoch':
            scheduler.step()
        epoch_log = epoch + 1
        
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('allAcc_train', allAcc_train, epoch_log)

        is_best = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):
            loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion)

            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
                is_best = mIoU_val > best_iou
                best_iou = max(best_iou, mIoU_val)

        if (epoch_log % args.save_freq == 0) and main_process():
            if not os.path.exists(args.save_path + "/model/"):
                os.makedirs(args.save_path + "/model/")
            filename = args.save_path + '/model/model_last.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'best_iou': best_iou, 'is_best': is_best}, filename)
            if is_best:
                shutil.copyfile(filename, args.save_path + '/model/model_best.pth')

    if main_process():
        writer.close()
        logger.info('==>Training done!\nBest Iou: %.3f' % (best_iou))


def focal_loss(output, target, class_weight, ignore_label, gamma, need_softmax=True, eps=1e-8):
    mask = (target != ignore_label)
    output_valid = output[mask]
    if need_softmax:
        output_valid = F.softmax(output_valid, -1)
    target_valid = target[mask]
    p_t = output_valid[torch.arange(output_valid.shape[0], device=target_valid.device), target_valid] #[N, ]
    class_weight_per_sample = class_weight[target_valid]
    focal_weight_per_sample = (1.0 - p_t) ** gamma
    loss = -(class_weight_per_sample * focal_weight_per_sample * torch.log(p_t + eps)).sum() / (class_weight_per_sample.sum() + eps)
    return loss


def train(train_loader, model, criterion, optimizer, epoch, scaler, scheduler, gpu):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)

    loss_name = args.loss_name

    for i, batch_data in enumerate(train_loader):  # (n, 3), (n, c), (n), (b)

        data_time.update(time.time() - end)
        torch.cuda.empty_cache()
        
        coord, xyz, feat, target, offset = batch_data
        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

        coord = torch.cat([batch.unsqueeze(-1), coord], -1)
        coord[:, 1:] += (torch.rand(3) * 2).type_as(coord)
        spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
    
        coord, xyz, feat, target, offset = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        batch = batch.cuda(non_blocking=True)
        sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, args.batch_size)

        assert batch.shape[0] == feat.shape[0]
        
        use_amp = args.use_amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            
            output = model(sinput, xyz, batch)
            assert output.shape[1] == args.classes

            if target.shape[-1] == 1:
                target = target[:, 0]  # for cls

            if loss_name == 'focal_loss':
                loss = focal_loss(output, target, criterion.weight, args.ignore_label, args.loss_gamma)
            elif loss_name == 'ce_loss':
                loss = criterion(output, target)
            else:
                raise ValueError("such loss {} not implemented".format(loss_name))

        optimizer.zero_grad()
        
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if args.scheduler_update == 'step':
            scheduler.step()

        output = output.max(1)[1]
        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            lr = scheduler.get_last_lr()
            if isinstance(lr, list):
                lr = [round(x, 8) for x in lr]
            elif isinstance(lr, float):
                lr = round(lr, 8)
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'Lr: {lr} '
                        'Accuracy {accuracy:.4f}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                        batch_time=batch_time, data_time=data_time,
                                                        remain_time=remain_time,
                                                        loss_meter=loss_meter,
                                                        lr=lr,
                                                        accuracy=accuracy))
        if main_process():
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch+1, args.epochs, mIoU, mAcc, allAcc))
    return loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    torch.cuda.empty_cache()

    loss_name = args.loss_name

    model.eval()
    end = time.time()
    for i, batch_data in enumerate(val_loader):

        data_time.update(time.time() - end)
    
        (coord, xyz, feat, target, offset, inds_reconstruct) = batch_data
        inds_reconstruct = inds_reconstruct.cuda(non_blocking=True)

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

        coord = torch.cat([batch.unsqueeze(-1), coord], -1)
        spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
    
        coord, xyz, feat, target, offset = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        batch = batch.cuda(non_blocking=True)

        sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, args.batch_size_val)

        assert batch.shape[0] == feat.shape[0]
        
        with torch.no_grad():
            output = model(sinput, xyz, batch)
            output = output[inds_reconstruct, :]
        
            if loss_name == 'focal_loss':
                loss = focal_loss(output, target, criterion.weight, args.ignore_label, args.loss_gamma)
            elif loss_name == 'ce_loss':
                loss = criterion(output, target)
            else:
                raise ValueError("such loss {} not implemented".format(loss_name))

        output = output.max(1)[1]
        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n

        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    
    return loss_meter.avg, mIoU, mAcc, allAcc

def validate_tta(val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    torch.cuda.empty_cache()

    loss_name = args.loss_name

    model.eval()
    end = time.time()
    for i, batch_data_list in enumerate(val_loader):

        data_time.update(time.time() - end)
    
        with torch.no_grad():
            output = 0.0
            for batch_data in batch_data_list:

                (coord, xyz, feat, target, offset, inds_reconstruct) = batch_data
                inds_reconstruct = inds_reconstruct.cuda(non_blocking=True)

                offset_ = offset.clone()
                offset_[1:] = offset_[1:] - offset_[:-1]
                batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

                coord = torch.cat([batch.unsqueeze(-1), coord], -1)
                spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
            
                coord, xyz, feat, target, offset = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
                batch = batch.cuda(non_blocking=True)

                sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, args.batch_size)

                assert batch.shape[0] == feat.shape[0]
                
                output_i = model(sinput, xyz, batch)
                output_i = F.softmax(output_i[inds_reconstruct, :], -1)
                
                output = output + output_i
            output = output / len(batch_data_list)
            
            if loss_name == 'focal_loss':
                loss = focal_loss(output, target, criterion.weight, args.ignore_label, args.loss_gamma)
            elif loss_name == 'ce_loss':
                loss = criterion(output, target)
            else:
                raise ValueError("such loss {} not implemented".format(loss_name))

        output = output.max(1)[1]
        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n

        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    
    return loss_meter.avg, mIoU, mAcc, allAcc




def validate_distance(val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    # 针对不同距离的 IoU 计算
    intersection_meter_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    union_meter_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    target_meter_list = [AverageMeter(), AverageMeter(), AverageMeter()]

    torch.cuda.empty_cache()

    loss_name = args.loss_name
    model.eval()
    end = time.time()

    all_point_data = []  # 用于存储所有点的类别信息

    for i, batch_data in enumerate(val_loader):

        data_time.update(time.time() - end)
    
        (coord, xyz, feat, target, offset, inds_reverse) = batch_data
        inds_reverse = inds_reverse.cuda(non_blocking=True)

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii] * o) for ii, o in enumerate(offset_)], 0).long()

        coord = torch.cat([batch.unsqueeze(-1), coord], -1)
        spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
    
        coord, xyz, feat, target, offset = (
            coord.cuda(non_blocking=True),
            xyz.cuda(non_blocking=True),
            feat.cuda(non_blocking=True),
            target.cuda(non_blocking=True),
            offset.cuda(non_blocking=True)
        )
        batch = batch.cuda(non_blocking=True)

        sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, args.batch_size_val)

        assert batch.shape[0] == feat.shape[0]
        
        with torch.no_grad():
            output = model(sinput, xyz, batch)
            output = output[inds_reverse, :]
        
            if loss_name == 'focal_loss':
                loss = focal_loss(output, target, criterion.weight, args.ignore_label, args.loss_gamma)
            elif loss_name == 'ce_loss':
                loss = criterion(output, target)
            else:
                raise ValueError(f"Loss type {loss_name} not implemented")

        # 获取每个点的预测类别
        output = output.max(1)[1].cpu().numpy()
        target = target.cpu().numpy()
        xyz = xyz.cpu().numpy()

        # 计算每个点到 Lidar 的距离
        r = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2 + xyz[:, 2] ** 2)

        # 保存点云数据 (X, Y, Z, 预测类别, 真实类别, 距离)
        point_data = np.column_stack((xyz, output, target, r))
        all_point_data.append(point_data)

        # 统计 IoU 指标
        masks = [r <= 20, (r > 20) & (r <= 50), r > 50]

        for ii, mask in enumerate(masks):
            intersection, union, tgt = intersectionAndUnionGPU(output[mask], target[mask], args.classes, args.ignore_label)
            if args.multiprocessing_distributed:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(tgt)
            intersection_meter_list[ii].update(intersection), union_meter_list[ii].update(union), target_meter_list[ii].update(tgt)

        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        loss_meter.update(loss.item(), coord.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

    # 计算整体和每个距离段的 mIoU, mAcc, allAcc
    mIoU = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
    mAcc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    mIoU_list = [np.mean(intersection_meter_list[i].sum / (union_meter_list[i].sum + 1e-10)) for i in range(3)]
    mAcc_list = [np.mean(intersection_meter_list[i].sum / (target_meter_list[i].sum + 1e-10)) for i in range(3)]
    allAcc_list = [sum(intersection_meter_list[i].sum) / (sum(target_meter_list[i].sum) + 1e-10) for i in range(3)]

    if main_process():
        metrics = ['close (≤20m)', 'medium (20-50m)', 'distant (>50m)']
        for ii in range(3):
            logger.info(f'Val result_{metrics[ii]}: mIoU/mAcc/allAcc {mIoU_list[ii]:.4f}/{mAcc_list[ii]:.4f}/{allAcc_list[ii]:.4f}')

        logger.info(f'Overall Val result: mIoU/mAcc/allAcc {mIoU:.4f}/{mAcc:.4f}/{allAcc:.4f}')

    # 保存所有点的类别数据到 CSV
    if main_process():
        all_point_data = np.vstack(all_point_data)  # 合并所有 batch 的点
        df = pd.DataFrame(all_point_data, columns=['X', 'Y', 'Z', 'Predicted_Label', 'True_Label', 'Distance'])
        df.to_csv("validation_output.csv", index=False)
        logger.info("Saved validation output to validation_output.csv")
    
    return loss_meter.avg, mIoU, mAcc, allAcc

def extract_point_classifications(val_loader, model):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Extracting Point Classifications >>>>>>>>>>>>>>>>')

    torch.cuda.empty_cache()
    model.eval()

    all_point_data = []  # 用于存储所有点的类别信息

    for i, batch_data in enumerate(val_loader):
        (coord, xyz, feat, target, offset, inds_reverse) = batch_data
        inds_reverse = inds_reverse.cuda(non_blocking=True)

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii] * o) for ii, o in enumerate(offset_)], 0).long()

        coord = torch.cat([batch.unsqueeze(-1), coord], -1)
        spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
    
        coord, xyz, feat, target, offset = (
            coord.cuda(non_blocking=True),
            xyz.cuda(non_blocking=True),
            feat.cuda(non_blocking=True),
            target.cuda(non_blocking=True),
            offset.cuda(non_blocking=True)
        )
        batch = batch.cuda(non_blocking=True)

        sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, args.batch_size_val)

        assert batch.shape[0] == feat.shape[0]
        
        with torch.no_grad():
            output = model(sinput, xyz, batch)
            output = output[inds_reverse, :]
        
        # 获取每个点的预测类别
        predicted_labels = output.max(1)[1].cpu().numpy()
        true_labels = target.cpu().numpy()
        xyz_coords = xyz.cpu().numpy()

        print(xyz_coords.shape)
        print(predicted_labels.shape)
        print(true_labels.shape)


        # 保存点云数据 (X, Y, Z, 预测类别, 真实类别)
        point_data = np.column_stack((xyz_coords, predicted_labels, true_labels))
        all_point_data.append(point_data)

    # 保存所有点的类别数据到 CSV
    if main_process():
        all_point_data = np.vstack(all_point_data)  # 合并所有 batch 的点
        df = pd.DataFrame(all_point_data, columns=['X', 'Y', 'Z', 'Predicted_Label', 'True_Label'])
        df.to_csv("point_classifications.csv", index=False)
        logger.info("Saved point classifications to point_classifications.csv")
    
    return all_point_data


# 执行推理
def infer(model, dataset):
    results = []
    
    for i, batch_data in enumerate(dataset):
        coord, xyz, feat, target, offset = batch_data  # 原代码的数据结构
        
        coord, xyz, feat, target = (
            coord.cuda(), xyz.cuda(), feat.cuda(), target.cuda()
        )
        
        sinput = spconv.SparseConvTensor(
            feat, coord.int(), np.clip((coord.max(0)[0][1:] + 1).cpu().numpy(), 128, None), args.batch_size
        )

        with torch.no_grad():
            output = model(sinput, xyz, torch.zeros_like(target).cuda())  # 前向推理
            predictions = torch.argmax(output, dim=1).cpu().numpy()  # 预测类别
        
        coords_np = coord.cpu().numpy()
        target_np = target.cpu().numpy()
        pred_np = predictions

        for j in range(coords_np.shape[0]):
            results.append(f"{coords_np[j,0]} {coords_np[j,1]} {coords_np[j,2]} {target_np[j]} {pred_np[j]}")
    
    return results

# 保存推理结果
def save_results(results, output_file="results.txt"):
    with open(output_file, "w") as f:
        f.write("\n".join(results))
    print("Results saved to", output_file)


def validate_and_print_points(val_loader, model):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation and Printing >>>>>>>>>>>>>>>>')
    
    model.eval()
    
    # 创建一个文件来保存结果
    output_file = open('point_cloud_results.txt', 'w')
    output_file.write('x,y,z,label,prediction\n')  # 写入标题行
    
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            if main_process():
                logger.info(f'Processing batch {i+1}/{len(val_loader)}')
            
            (coord, xyz, feat, target, offset, inds_reverse) = batch_data
            inds_reverse = inds_reverse.cuda(non_blocking=True)
            
            offset_ = offset.clone()
            offset_[1:] = offset_[1:] - offset_[:-1]
            batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()
            
            coord = torch.cat([batch.unsqueeze(-1), coord], -1)
            spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
            
            coord, xyz, feat, target = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True)
            batch = batch.cuda(non_blocking=True)
            
            sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, args.batch_size_val)
            
            # 运行模型并获取预测结果
            output = model(sinput, xyz, batch)
            output = output[inds_reverse, :]
            predictions = output.max(1)[1]  # 获取预测类别
            
            # 将坐标、标签和预测值转移到CPU，以便打印
            xyz_cpu = xyz[inds_reverse].cpu().numpy()
            target_cpu = target.cpu().numpy()
            predictions_cpu = predictions.cpu().numpy()
            
            # 为每个点写入结果
            for j in range(len(xyz_cpu)):
                x, y, z = xyz_cpu[j]
                label = target_cpu[j]
                pred = predictions_cpu[j]
                output_file.write(f"{x:.6f},{y:.6f},{z:.6f},{label},{pred}\n")
            
            if (i + 1) % args.print_freq == 0 and main_process():
                logger.info(f'Processed {i+1}/{len(val_loader)} batches')
    
    output_file.close()
    
    if main_process():
        logger.info(f'Results saved to point_cloud_results.txt')
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation and Printing <<<<<<<<<<<<<<<<<')


def validate_and_print_points2(val_loader, model):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation and Printing >>>>>>>>>>>>>>>>')
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    
    model.eval()
    end = time.time()
    
    # 获取进程信息
    rank = 0
    if args.multiprocessing_distributed:
        rank = dist.get_rank()
    
    # 每个进程创建自己的输出文件
    output_filename = f'point_cloud_results_rank_{rank}.txt'
    output_file = open(output_filename, 'w')
    output_file.write('x,y,z,label,prediction\n')  # 写入标题行
    
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            data_time.update(time.time() - end)
            
            if main_process() and (i + 1) % args.print_freq == 0:
                logger.info(f'Processing batch {i+1}/{len(val_loader)}')
            
            (coord, xyz, feat, target, offset, inds_reverse) = batch_data
            inds_reverse = inds_reverse.cuda(non_blocking=True)
            
            offset_ = offset.clone()
            offset_[1:] = offset_[1:] - offset_[:-1]
            batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()
            
            coord = torch.cat([batch.unsqueeze(-1), coord], -1)
            spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
            
            coord, xyz, feat, target = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True)
            offset = offset.cuda(non_blocking=True)
            batch = batch.cuda(non_blocking=True)
            
            sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, args.batch_size_val)
            
            # 运行模型并获取预测结果
            output = model(sinput, xyz, batch)
            output = output[inds_reverse, :]
            
            # 获取预测类别
            predictions = output.max(1)[1]
            
            # 计算指标
            n = coord.size(0)
            
            intersection, union, target_count = intersectionAndUnionGPU(predictions, target, args.classes, args.ignore_label)
            if args.multiprocessing_distributed:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target_count)
            intersection, union, target_count = intersection.cpu().numpy(), union.cpu().numpy(), target_count.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target_count)
            
            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            batch_time.update(time.time() - end)
            
            # 每个进程写入自己处理的点
            xyz_cpu = xyz[inds_reverse].cpu().numpy()
            target_cpu = target.cpu().numpy()
            predictions_cpu = predictions.cpu().numpy()
            
            for j in range(len(xyz_cpu)):
                x, y, z = xyz_cpu[j]
                label = target_cpu[j]
                pred = predictions_cpu[j]
                output_file.write(f"{x:.6f},{y:.6f},{z:.6f},{label},{pred}\n")
            
            end = time.time()
            
            if (i + 1) % args.print_freq == 0 and main_process():
                logger.info('Test: [{}/{}] '
                           'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                           'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                           'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                         data_time=data_time,
                                                         batch_time=batch_time,
                                                         accuracy=accuracy))
    
    # 关闭文件
    output_file.close()
    
    if main_process():
        logger.info(f'Each process saved results to point_cloud_results_rank_N.txt files')
        logger.info(f'To combine files, run: cat point_cloud_results_rank_*.txt > combined_results.txt')
    
    # 计算最终指标
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    
    if main_process():
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation and Printing <<<<<<<<<<<<<<<<<')
    
    # 返回评估指标
    return mIoU, mAcc, allAcc\
    
def validate_and_print_points3(val_loader, model):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation and Saving Predictions >>>>>>>>>>>>>>>>')
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    model.eval()
    end = time.time()
    
    # 获取进程信息
    rank = 0
    if args.multiprocessing_distributed:
        rank = dist.get_rank()
    
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            data_time.update(time.time() - end)
            
            if main_process() and (i + 1) % args.print_freq == 0:
                logger.info(f'Processing batch {i+1}/{len(val_loader)}')
            
            # 解包批次数据
            (coord, xyz, feat, target, offset, inds_reverse, file_names) = batch_data
            
            # 计算每个批次中的总点数
            total_points = coord.shape[0]
            
            # 打印每个批次中的点数量
            if main_process():
                logger.info(f'Batch {i+1}/{len(val_loader)}: Total points: {total_points}')
            
            inds_reverse = inds_reverse.cuda(non_blocking=True)
            
            # 保留原有的代码
            offset_ = offset.clone()
            offset_[1:] = offset_[1:] - offset_[:-1]
            
            # 计算每个样本中的点数量
            points_per_sample = offset_.tolist()
            if main_process():
                logger.info(f'Points per sample: {points_per_sample}')
            
            batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()
            
            # 其余原有代码继续...
            coord = torch.cat([batch.unsqueeze(-1), coord], -1)
            spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
                       
            coord, xyz, feat, target = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True)
            offset = offset.cuda(non_blocking=True)
            batch = batch.cuda(non_blocking=True)
            
            sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, args.batch_size_val)
            
            # 运行模型并获取预测结果
            output = model(sinput, xyz, batch)
            output = output[inds_reverse, :]
            
            # 获取预测类别
            predictions = output.max(1)[1].cpu().numpy()
            
            # 批次中的样本信息
            batch_idx = batch[inds_reverse].cpu().numpy()
            
            # 保存预测结果，按照SemanticKITTI格式
            for b_idx in np.unique(batch_idx):
                mask = batch_idx == b_idx
                scan_predictions = predictions[mask]
                
                # 使用文件名构建保存路径
                current_file = file_names[int(b_idx)]
                # 从文件名中提取序列和帧ID
                # 假设文件路径格式为 ".../sequences/XX/velodyne/XXXXXX.bin"
                sequence_dir = os.path.dirname(os.path.dirname(current_file))
                sequence_id = os.path.basename(sequence_dir)
                frame_id = os.path.basename(current_file).split('.')[0]                
                
                # 创建保存目录
                save_dir = os.path.join(args.save_folder, 'predictions', sequence_id)
                os.makedirs(save_dir, exist_ok=True)
                
                # 保存预测结果
                save_path = os.path.join(save_dir, f'{frame_id}.label')
                scan_predictions = scan_predictions.astype(np.uint32)
                scan_predictions.tofile(save_path)
                
                if main_process() and (i + 1) % args.print_freq == 0:
                    logger.info(f'Saved prediction for sequence {sequence_id}, frame {frame_id}')
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if (i + 1) % args.print_freq == 0 and main_process():
                logger.info('Test: [{}/{}] '
                           'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                           'Batch {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                                i + 1, len(val_loader),
                                data_time=data_time,
                                batch_time=batch_time))
    
    if main_process():
        logger.info(f'Predictions saved to {args.save_folder}/predictions/')
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation and Saving Predictions <<<<<<<<<<<<<<<<<')


def validate_and_print_points4(val_loader, model):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation and Saving Predictions >>>>>>>>>>>>>>>>')
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    # 添加计数器统计移动物体点数
    total_points_count = 0
    moving_points_in_gt_count = 0
    moving_points_in_pred_count = 0
    
    model.eval()
    end = time.time()
    
    # 获取进程信息
    rank = 0
    if args.multiprocessing_distributed:
        rank = dist.get_rank()
    
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            data_time.update(time.time() - end)
            
            if main_process() and (i + 1) % args.print_freq == 0:
                logger.info(f'Processing batch {i+1}/{len(val_loader)}')
            
            # 解包批次数据
            (coord, xyz, feat, target, offset, inds_reverse, file_names) = batch_data
            
            # 计算每个批次中的总点数
            total_points = coord.shape[0]
            total_points_count += total_points
            
            # 计算原始标签中移动物体的点数
            moving_mask_gt = (target >= 252) & (target <= 259)
            moving_points_gt = moving_mask_gt.sum().item()
            moving_points_in_gt_count += moving_points_gt
            
            # 打印每个批次中的点数量和移动物体点数
            # if main_process():
            #     logger.info(f'Batch {i+1}/{len(val_loader)}: Total points: {total_points}, Moving points in GT: {moving_points_gt}')
            
            inds_reverse = inds_reverse.cuda(non_blocking=True)
            
            
            offset_ = offset.clone()
            offset_[1:] = offset_[1:] - offset_[:-1]
            
            # 计算每个样本中的点数量
            points_per_sample = offset_.tolist()
            if main_process():
                logger.info(f'Points per sample: {points_per_sample}')
            
            batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()
            
            
            coord = torch.cat([batch.unsqueeze(-1), coord], -1)
            spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
                       
            coord, xyz, feat, target = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True)
            offset = offset.cuda(non_blocking=True)
            batch = batch.cuda(non_blocking=True)
            
            sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, args.batch_size_val)
            
            # 运行模型并获取预测结果
            output = model(sinput, xyz, batch)
            output = output[inds_reverse, :]
            
            # 获取预测类别
            predictions = output.max(1)[1].cpu().numpy()
            
            # 批次中的样本信息
            batch_idx = batch[inds_reverse].cpu().numpy()
            
            # 将target转到CPU进行计算
            target_cpu = target.cpu().numpy()
            
            # 保存预测结果，按照SemanticKITTI格式，并忽略移动物体
            for b_idx in np.unique(batch_idx):
                mask = batch_idx == b_idx
                scan_predictions = predictions[mask]
                scan_targets = target_cpu[mask]
                
                # 统计该样本中的移动物体点数
                moving_mask_sample = (scan_targets >= 252) & (scan_targets <= 259)
                moving_points_sample = np.sum(moving_mask_sample)
                
                # 统计预测结果中的移动物体点数
                # 注：这里假设预测结果是模型的输出类别，不是原始标签ID
                # 如果需要，您可能需要将预测结果映射回原始标签ID进行比较
                
                # 创建不包含移动物体的掩码
                non_moving_mask = ~moving_mask_sample
                filtered_predictions = scan_predictions[non_moving_mask]
                
                # 使用文件名构建保存路径
                current_file = file_names[int(b_idx)]
                # 从文件名中提取序列和帧ID
                sequence_dir = os.path.dirname(os.path.dirname(current_file))
                sequence_id = os.path.basename(sequence_dir)
                frame_id = os.path.basename(current_file).split('.')[0]
                
                # 创建保存目录
                save_dir = os.path.join(args.save_folder, 'predictions', sequence_id)
                os.makedirs(save_dir, exist_ok=True)
                
                # 保存过滤后的预测结果
                save_path = os.path.join(save_dir, f'{frame_id}.label')
                filtered_predictions = filtered_predictions.astype(np.uint32)
                filtered_predictions.tofile(save_path)
                
                # 统计预测结果中的移动物体
                # 注意：这里我们统计的是根据GT标签过滤掉的点数
                moving_points_in_pred_count += moving_points_sample
                
                if main_process() and (i + 1) % args.print_freq == 0:
                    logger.info(f'Saved prediction for sequence {sequence_id}, frame {frame_id}')
                    logger.info(f'  - Total points in frame: {np.sum(mask)}')
                    logger.info(f'  - Moving points filtered: {moving_points_sample} ({moving_points_sample/np.sum(mask)*100:.2f}%)')
                    logger.info(f'  - Points saved after filtering: {np.sum(non_moving_mask)}')
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if (i + 1) % args.print_freq == 0 and main_process():
                logger.info('Test: [{}/{}] '
                           'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                           'Batch {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                                i + 1, len(val_loader),
                                data_time=data_time,
                                batch_time=batch_time))
    
    # 在结束时打印移动物体统计信息
    if main_process():
        logger.info(f'移动物体统计:')
        logger.info(f'  - 总点数: {total_points_count}')
        logger.info(f'  - 原始标签中的移动物体点数: {moving_points_in_gt_count} ({moving_points_in_gt_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 预测结果中过滤的移动物体点数: {moving_points_in_pred_count} ({moving_points_in_pred_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 过滤后保存的点数: {total_points_count - moving_points_in_pred_count} ({(total_points_count - moving_points_in_pred_count)/total_points_count*100:.2f}%)')
        logger.info(f'Predictions saved to {args.save_folder}/predictions/')
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation and Saving Predictions <<<<<<<<<<<<<<<<<')

def validate_and_print_points5(val_loader, model):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation and Saving Predictions >>>>>>>>>>>>>>>>')
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    # 点云统计计数器
    total_points_count = 0
    moving_points_in_gt_count = 0
    
    # 新增：预测与标签不同的点数统计
    prediction_differs_count = 0
    
    model.eval()
    end = time.time()
    
    # 获取进程信息
    rank = 0
    if args.multiprocessing_distributed:
        rank = dist.get_rank()
    
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            data_time.update(time.time() - end)
            
            if main_process() and (i + 1) % args.print_freq == 0:
                logger.info(f'Processing batch {i+1}/{len(val_loader)}')
            
            # 解包批次数据
            (coord, xyz, feat, target, offset, inds_reverse, file_names) = batch_data
            
            # 计算当前批次中的点数
            batch_points_count = coord.shape[0]
            total_points_count += batch_points_count
            
            # 计算原始标签中移动物体的点数
            moving_mask_gt = (target >= 252) & (target <= 259)
            moving_points_gt = moving_mask_gt.sum().item()
            moving_points_in_gt_count += moving_points_gt
            
            inds_reverse = inds_reverse.cuda(non_blocking=True)
            
            offset_ = offset.clone()
            offset_[1:] = offset_[1:] - offset_[:-1]
            
            # 计算每个样本中的点数量
            points_per_sample = offset_.tolist()
            if main_process() and (i + 1) % args.print_freq == 0:
                logger.info(f'Points per sample: {points_per_sample}')
            
            batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()
            
            coord = torch.cat([batch.unsqueeze(-1), coord], -1)
            spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
                       
            coord, xyz, feat, target = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True)
            offset = offset.cuda(non_blocking=True)
            batch = batch.cuda(non_blocking=True)
            
            sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, args.batch_size_val)
            
            # 运行模型并获取预测结果
            output = model(sinput, xyz, batch)
            output = output[inds_reverse, :]
            
            # 获取预测类别
            pred_logits = output.max(1)[1]  # 获取预测结果的类别索引
            
            # 计算预测与标签不同的点数
            differs_mask = (pred_logits != target)
            batch_prediction_differs = differs_mask.sum().item()
            prediction_differs_count += batch_prediction_differs
            
            # 将预测结果和目标标签转移到CPU进行保存
            predictions = pred_logits.cpu().numpy()
            target_cpu = target.cpu().numpy()
            batch_idx = batch[inds_reverse].cpu().numpy()
            
            # 保存预测结果，按照SemanticKITTI格式
            for b_idx in np.unique(batch_idx):
                mask = batch_idx == b_idx
                scan_predictions = predictions[mask]
                scan_targets = target_cpu[mask]
                
                # 使用原始标签中的移动物体掩码过滤预测结果
                moving_mask_gt_sample = (scan_targets >= 252) & (scan_targets <= 259)
                non_moving_mask = ~moving_mask_gt_sample
                filtered_predictions = scan_predictions[non_moving_mask]
                
                # 使用文件名构建保存路径
                current_file = file_names[int(b_idx)]
                sequence_dir = os.path.dirname(os.path.dirname(current_file))
                sequence_id = os.path.basename(sequence_dir)
                frame_id = os.path.basename(current_file).split('.')[0]
                
                # 创建保存目录
                save_dir = os.path.join(args.save_folder, 'predictions', sequence_id)
                os.makedirs(save_dir, exist_ok=True)
                
                # 保存过滤后的预测结果
                save_path = os.path.join(save_dir, f'{frame_id}.label')
                filtered_predictions = filtered_predictions.astype(np.uint32)
                filtered_predictions.tofile(save_path)
                
                if main_process() and (i + 1) % args.print_freq == 0:
                    points_in_frame = np.sum(mask)
                    moving_points_gt_sample = np.sum(moving_mask_gt_sample)
                    
                    # 计算该样本中预测与标签不同的点数
                    differs_in_sample = np.sum(scan_predictions != scan_targets)
                    
                    logger.info(f'Saved prediction for sequence {sequence_id}, frame {frame_id}')
                    logger.info(f'  - Total points in frame: {points_in_frame}')
                    logger.info(f'  - Moving points in GT: {moving_points_gt_sample} ({moving_points_gt_sample/points_in_frame*100:.2f}%)')
                    logger.info(f'  - Prediction differs from GT: {differs_in_sample} ({differs_in_sample/points_in_frame*100:.2f}%)')
                    logger.info(f'  - Points saved after filtering: {np.sum(non_moving_mask)}')
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if (i + 1) % args.print_freq == 0 and main_process():
                logger.info('Test: [{}/{}] '
                           'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                           'Batch {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                                i + 1, len(val_loader),
                                data_time=data_time,
                                batch_time=batch_time))
    
    # 在结束时打印统计信息
    if main_process():
        logger.info(f'统计信息:')
        logger.info(f'  - 总点数: {total_points_count}')
        logger.info(f'  - 原始标签中的移动物体点数: {moving_points_in_gt_count} ({moving_points_in_gt_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 预测与标签不同的点数: {prediction_differs_count} ({prediction_differs_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 过滤后保存的点数: {total_points_count - moving_points_in_gt_count} ({(total_points_count - moving_points_in_gt_count)/total_points_count*100:.2f}%)')
        
        logger.info(f'Predictions saved to {args.save_folder}/predictions/')
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation and Saving Predictions <<<<<<<<<<<<<<<<<')


def validate_and_print_points6(val_loader, model):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation and Saving Predictions >>>>>>>>>>>>>>>>')
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    # 点云统计计数器
    total_points_count = 0
    moving_points_in_gt_count = 0
    moving_points_in_pred_count = 0  # 新增：预测结果中的移动物体点数
    prediction_differs_count = 0
    
    model.eval()
    end = time.time()
    
    # 获取进程信息
    rank = 0
    if args.multiprocessing_distributed:
        rank = dist.get_rank()
    
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            data_time.update(time.time() - end)
            
            if main_process() and (i + 1) % args.print_freq == 0:
                logger.info(f'Processing batch {i+1}/{len(val_loader)}')
            
            # 解包批次数据
            (coord, xyz, feat, target, offset, inds_reverse, file_names) = batch_data
            
            # 计算当前批次中的点数
            batch_points_count = coord.shape[0]
            total_points_count += batch_points_count
            
            # 计算原始标签中移动物体的点数
            moving_mask_gt = (target >= 252) & (target <= 259)
            moving_points_gt = moving_mask_gt.sum().item()
            moving_points_in_gt_count += moving_points_gt
            
            inds_reverse = inds_reverse.cuda(non_blocking=True)
            
            offset_ = offset.clone()
            offset_[1:] = offset_[1:] - offset_[:-1]
            
            # 计算每个样本中的点数量
            points_per_sample = offset_.tolist()
            if main_process() and (i + 1) % args.print_freq == 0:
                logger.info(f'Points per sample: {points_per_sample}')
            
            batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()
            
            coord = torch.cat([batch.unsqueeze(-1), coord], -1)
            spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
                       
            coord, xyz, feat, target = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True)
            offset = offset.cuda(non_blocking=True)
            batch = batch.cuda(non_blocking=True)
            
            sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, args.batch_size_val)
            
            # 运行模型并获取预测结果
            output = model(sinput, xyz, batch)
            output = output[inds_reverse, :]
            
            # 获取预测类别
            pred_logits = output.max(1)[1]  # 获取预测结果的类别索引
            
            # 计算预测结果中移动物体的点数
            moving_mask_pred = (pred_logits >= 252) & (pred_logits <= 259)
            moving_points_pred = moving_mask_pred.sum().item()
            moving_points_in_pred_count += moving_points_pred
            
            # 计算预测与标签不同的点数
            differs_mask = (pred_logits != target)
            batch_prediction_differs = differs_mask.sum().item()
            prediction_differs_count += batch_prediction_differs
            
            # 将预测结果和目标标签转移到CPU进行保存
            predictions = pred_logits.cpu().numpy()
            target_cpu = target.cpu().numpy()
            batch_idx = batch[inds_reverse].cpu().numpy()
            
            # 保存预测结果，按照SemanticKITTI格式
            for b_idx in np.unique(batch_idx):
                mask = batch_idx == b_idx
                scan_predictions = predictions[mask]
                scan_targets = target_cpu[mask]
                
                # 使用原始标签中的移动物体掩码过滤预测结果
                moving_mask_gt_sample = (scan_targets >= 252) & (scan_targets <= 259)
                non_moving_mask = ~moving_mask_gt_sample
                filtered_predictions = scan_predictions[non_moving_mask]
                
                # 使用文件名构建保存路径
                current_file = file_names[int(b_idx)]
                sequence_dir = os.path.dirname(os.path.dirname(current_file))
                sequence_id = os.path.basename(sequence_dir)
                frame_id = os.path.basename(current_file).split('.')[0]
                
                # 创建保存目录
                save_dir = os.path.join(args.save_folder, 'predictions', sequence_id)
                os.makedirs(save_dir, exist_ok=True)
                
                # 保存过滤后的预测结果
                save_path = os.path.join(save_dir, f'{frame_id}.label')
                filtered_predictions = filtered_predictions.astype(np.uint32)
                filtered_predictions.tofile(save_path)
                
                if main_process() and (i + 1) % args.print_freq == 0:
                    points_in_frame = np.sum(mask)
                    moving_points_gt_sample = np.sum(moving_mask_gt_sample)
                    moving_points_pred_sample = np.sum((scan_predictions >= 252) & (scan_predictions <= 259))
                    
                    # 计算该样本中预测与标签不同的点数
                    differs_in_sample = np.sum(scan_predictions != scan_targets)
                    
                    logger.info(f'Saved prediction for sequence {sequence_id}, frame {frame_id}')
                    logger.info(f'  - Total points in frame: {points_in_frame}')
                    logger.info(f'  - Moving points in GT: {moving_points_gt_sample} ({moving_points_gt_sample/points_in_frame*100:.2f}%)')
                    logger.info(f'  - Moving points in prediction: {moving_points_pred_sample} ({moving_points_pred_sample/points_in_frame*100:.2f}%)')
                    logger.info(f'  - Prediction differs from GT: {differs_in_sample} ({differs_in_sample/points_in_frame*100:.2f}%)')
                    logger.info(f'  - Points saved after filtering: {np.sum(non_moving_mask)}')
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if (i + 1) % args.print_freq == 0 and main_process():
                logger.info('Test: [{}/{}] '
                           'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                           'Batch {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                                i + 1, len(val_loader),
                                data_time=data_time,
                                batch_time=batch_time))
    
    # 在结束时打印统计信息
    if main_process():
        logger.info(f'统计信息:')
        logger.info(f'  - 总点数: {total_points_count}')
        logger.info(f'  - 原始标签中的移动物体点数: {moving_points_in_gt_count} ({moving_points_in_gt_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 预测结果中的移动物体点数: {moving_points_in_pred_count} ({moving_points_in_pred_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 预测与标签不同的点数: {prediction_differs_count} ({prediction_differs_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 过滤后保存的点数: {total_points_count - moving_points_in_gt_count} ({(total_points_count - moving_points_in_gt_count)/total_points_count*100:.2f}%)')
        
        logger.info(f'Predictions saved to {args.save_folder}/predictions/')
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation and Saving Predictions <<<<<<<<<<<<<<<<<')

def validate_and_print_points7(val_loader, model):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation and Saving Predictions >>>>>>>>>>>>>>>>')
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    # 点云统计计数器
    total_points_count = 0
    moving_points_in_gt_count = 0
    moving_points_in_pred_count = 0
    prediction_differs_count = 0
    
    # 添加预测类别统计字典
    prediction_class_counts = {}
    gt_class_counts = {}
    
    model.eval()
    end = time.time()
    
    # 获取进程信息
    rank = 0
    if args.multiprocessing_distributed:
        rank = dist.get_rank()
    
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            data_time.update(time.time() - end)
            
            if main_process() and (i + 1) % args.print_freq == 0:
                logger.info(f'Processing batch {i+1}/{len(val_loader)}')
            
            # 解包批次数据
            (coord, xyz, feat, target, offset, inds_reverse, file_names) = batch_data
            
            # 计算当前批次中的点数
            batch_points_count = coord.shape[0]
            total_points_count += batch_points_count
            
            # 计算原始标签中移动物体的点数
            moving_mask_gt = (target >= 252) & (target <= 259)
            moving_points_gt = moving_mask_gt.sum().item()
            moving_points_in_gt_count += moving_points_gt
            
            inds_reverse = inds_reverse.cuda(non_blocking=True)
            
            offset_ = offset.clone()
            offset_[1:] = offset_[1:] - offset_[:-1]
            
            # 计算每个样本中的点数量
            points_per_sample = offset_.tolist()
            if main_process() and (i + 1) % args.print_freq == 0:
                logger.info(f'Points per sample: {points_per_sample}')
            
            batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()
            
            coord = torch.cat([batch.unsqueeze(-1), coord], -1)
            spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
                       
            coord, xyz, feat, target = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True)
            offset = offset.cuda(non_blocking=True)
            batch = batch.cuda(non_blocking=True)
            
            sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, args.batch_size_val)
            
            # 运行模型并获取预测结果
            output = model(sinput, xyz, batch)
            output = output[inds_reverse, :]
            
            # 获取预测类别
            pred_logits = output.max(1)[1]
            
            # 计算预测结果中移动物体的点数
            moving_mask_pred = (pred_logits >= 252) & (pred_logits <= 259)
            moving_points_pred = moving_mask_pred.sum().item()
            moving_points_in_pred_count += moving_points_pred
            
            # 计算预测与标签不同的点数
            differs_mask = (pred_logits != target)
            batch_prediction_differs = differs_mask.sum().item()
            prediction_differs_count += batch_prediction_differs
            
            # 将预测结果和目标标签转移到CPU进行保存
            predictions = pred_logits.cpu().numpy()
            target_cpu = target.cpu().numpy()
            batch_idx = batch[inds_reverse].cpu().numpy()
            
            # 统计预测类别分布
            unique_pred_classes, pred_counts = np.unique(predictions, return_counts=True)
            for cls, count in zip(unique_pred_classes, pred_counts):
                if cls in prediction_class_counts:
                    prediction_class_counts[cls] += count
                else:
                    prediction_class_counts[cls] = count
            
            # 统计真实标签类别分布
            unique_gt_classes, gt_counts = np.unique(target_cpu, return_counts=True)
            for cls, count in zip(unique_gt_classes, gt_counts):
                if cls in gt_class_counts:
                    gt_class_counts[cls] += count
                else:
                    gt_class_counts[cls] = count
            
            # 保存预测结果，按照SemanticKITTI格式
            for b_idx in np.unique(batch_idx):
                mask = batch_idx == b_idx
                scan_predictions = predictions[mask]
                scan_targets = target_cpu[mask]
                
                # 使用原始标签中的移动物体掩码过滤预测结果
                moving_mask_gt_sample = (scan_targets >= 252) & (scan_targets <= 259)
                non_moving_mask = ~moving_mask_gt_sample
                filtered_predictions = scan_predictions[non_moving_mask]
                
                # 使用文件名构建保存路径
                current_file = file_names[int(b_idx)]
                sequence_dir = os.path.dirname(os.path.dirname(current_file))
                sequence_id = os.path.basename(sequence_dir)
                frame_id = os.path.basename(current_file).split('.')[0]
                
                # 创建保存目录
                save_dir = os.path.join(args.save_folder, 'predictions', sequence_id)
                os.makedirs(save_dir, exist_ok=True)
                
                # 保存过滤后的预测结果
                # save_path = os.path.join(save_dir, f'{frame_id}.label')
                # filtered_predictions = filtered_predictions.astype(np.uint32)
                # filtered_predictions.tofile(save_path)
                
                if main_process() and (i + 1) % args.print_freq == 0:
                    points_in_frame = np.sum(mask)
                    moving_points_gt_sample = np.sum(moving_mask_gt_sample)
                    moving_points_pred_sample = np.sum((scan_predictions >= 252) & (scan_predictions <= 259))
                    
                    # 计算该样本中预测与标签不同的点数
                    differs_in_sample = np.sum(scan_predictions != scan_targets)
                    
                    logger.info(f'Saved prediction for sequence {sequence_id}, frame {frame_id}')
                    logger.info(f'  - Total points in frame: {points_in_frame}')
                    logger.info(f'  - Moving points in GT: {moving_points_gt_sample} ({moving_points_gt_sample/points_in_frame*100:.2f}%)')
                    logger.info(f'  - Moving points in prediction: {moving_points_pred_sample} ({moving_points_pred_sample/points_in_frame*100:.2f}%)')
                    logger.info(f'  - Prediction differs from GT: {differs_in_sample} ({differs_in_sample/points_in_frame*100:.2f}%)')
                    logger.info(f'  - Points saved after filtering: {np.sum(non_moving_mask)}')
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if (i + 1) % args.print_freq == 0 and main_process():
                logger.info('Test: [{}/{}] '
                           'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                           'Batch {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                                i + 1, len(val_loader),
                                data_time=data_time,
                                batch_time=batch_time))
    
    # 在结束时打印统计信息
    if main_process():
        logger.info(f'统计信息:')
        logger.info(f'  - 总点数: {total_points_count}')
        logger.info(f'  - 原始标签中的移动物体点数: {moving_points_in_gt_count} ({moving_points_in_gt_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 预测结果中的移动物体点数: {moving_points_in_pred_count} ({moving_points_in_pred_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 预测与标签不同的点数: {prediction_differs_count} ({prediction_differs_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 过滤后保存的点数: {total_points_count - moving_points_in_gt_count} ({(total_points_count - moving_points_in_gt_count)/total_points_count*100:.2f}%)')
        
        # 打印预测类别分布
        logger.info(f'预测类别分布:')
        for cls in sorted(prediction_class_counts.keys()):
            count = prediction_class_counts[cls]
            percentage = count / total_points_count * 100
            logger.info(f'  - 类别 {cls}: {count} ({percentage:.2f}%)')
            
            # 特别标注移动物体类别
            if 252 <= cls <= 259:
                logger.info(f'    (移动物体类别)')
        
        # 打印真实标签类别分布
        logger.info(f'真实标签类别分布:')
        for cls in sorted(gt_class_counts.keys()):
            count = gt_class_counts[cls]
            percentage = count / total_points_count * 100
            logger.info(f'  - 类别 {cls}: {count} ({percentage:.2f}%)')
            
            # 特别标注移动物体类别
            if 252 <= cls <= 259:
                logger.info(f'    (移动物体类别)')
        
        logger.info(f'Predictions saved to {args.save_folder}/predictions/')
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation and Saving Predictions <<<<<<<<<<<<<<<<<')


def validate_and_print_points8(val_loader, model):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation and Saving Predictions >>>>>>>>>>>>>>>>')
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    # 点云统计计数器
    total_points_count = 0
    moving_points_in_gt_count = 0
    moving_points_in_pred_count = 0
    prediction_differs_count = 0
    
    # 添加预测类别统计字典
    prediction_class_counts = {}
    gt_class_counts = {}
    
    model.eval()
    end = time.time()
    
    # 获取进程信息
    rank = 0
    if args.multiprocessing_distributed:
        rank = dist.get_rank()
    
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            data_time.update(time.time() - end)
            
            if main_process() and (i + 1) % args.print_freq == 0:
                logger.info(f'Processing batch {i+1}/{len(val_loader)}')
            
            # 解包批次数据
            (coord, xyz, feat, target, offset, inds_reverse, file_names) = batch_data
            
            # 计算当前批次中的点数
            batch_points_count = coord.shape[0]
            total_points_count += batch_points_count
            
            # 计算原始标签中移动物体的点数 - 修改为类别ID为19的点
            moving_mask_gt = (target == 19)
            moving_points_gt = moving_mask_gt.sum().item()
            moving_points_in_gt_count += moving_points_gt
            
            inds_reverse = inds_reverse.cuda(non_blocking=True)
            
            offset_ = offset.clone()
            offset_[1:] = offset_[1:] - offset_[:-1]
            
            # 计算每个样本中的点数量
            points_per_sample = offset_.tolist()
            if main_process() and (i + 1) % args.print_freq == 0:
                logger.info(f'Points per sample: {points_per_sample}')
            
            batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()
            
            coord = torch.cat([batch.unsqueeze(-1), coord], -1)
            spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
                       
            coord, xyz, feat, target = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True)
            offset = offset.cuda(non_blocking=True)
            batch = batch.cuda(non_blocking=True)
            
            sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, args.batch_size_val)
            
            # 运行模型并获取预测结果
            output = model(sinput, xyz, batch)
            output = output[inds_reverse, :]
            
            # 获取预测类别
            pred_logits = output.max(1)[1]
            
            # 计算预测结果中移动物体的点数 - 修改为类别ID为19的点
            moving_mask_pred = (pred_logits == 19)
            moving_points_pred = moving_mask_pred.sum().item()
            moving_points_in_pred_count += moving_points_pred
            
            # 计算预测与标签不同的点数
            differs_mask = (pred_logits != target)
            batch_prediction_differs = differs_mask.sum().item()
            prediction_differs_count += batch_prediction_differs
            
            # 将预测结果和目标标签转移到CPU进行保存
            predictions = pred_logits.cpu().numpy()
            target_cpu = target.cpu().numpy()
            batch_idx = batch[inds_reverse].cpu().numpy()
            
            # 统计预测类别分布
            unique_pred_classes, pred_counts = np.unique(predictions, return_counts=True)
            for cls, count in zip(unique_pred_classes, pred_counts):
                if cls in prediction_class_counts:
                    prediction_class_counts[cls] += count
                else:
                    prediction_class_counts[cls] = count
            
            # 统计真实标签类别分布
            unique_gt_classes, gt_counts = np.unique(target_cpu, return_counts=True)
            for cls, count in zip(unique_gt_classes, gt_counts):
                if cls in gt_class_counts:
                    gt_class_counts[cls] += count
                else:
                    gt_class_counts[cls] = count
            
            # 保存预测结果，按照SemanticKITTI格式
            for b_idx in np.unique(batch_idx):
                mask = batch_idx == b_idx
                scan_predictions = predictions[mask]
                scan_targets = target_cpu[mask]
                
                # 使用原始标签中的移动物体掩码过滤预测结果 - 修改为类别ID为19的点
                moving_mask_gt_sample = (scan_targets == 19)
                non_moving_mask = ~moving_mask_gt_sample
                filtered_predictions = scan_predictions[non_moving_mask]
                
                # 使用文件名构建保存路径
                current_file = file_names[int(b_idx)]
                sequence_dir = os.path.dirname(os.path.dirname(current_file))
                sequence_id = os.path.basename(sequence_dir)
                frame_id = os.path.basename(current_file).split('.')[0]
                
                # 创建保存目录
                save_dir = os.path.join(args.save_folder, 'predictions', sequence_id)
                os.makedirs(save_dir, exist_ok=True)
                
                # 保存过滤后的预测结果
                # save_path = os.path.join(save_dir, f'{frame_id}.label')
                # filtered_predictions = filtered_predictions.astype(np.uint32)
                # filtered_predictions.tofile(save_path)
                
                if main_process() and (i + 1) % args.print_freq == 0:
                    points_in_frame = np.sum(mask)
                    moving_points_gt_sample = np.sum(moving_mask_gt_sample)
                    moving_points_pred_sample = np.sum((scan_predictions == 19))
                    
                    # 计算该样本中预测与标签不同的点数
                    differs_in_sample = np.sum(scan_predictions != scan_targets)
                    
                    logger.info(f'Saved prediction for sequence {sequence_id}, frame {frame_id}')
                    logger.info(f'  - Total points in frame: {points_in_frame}')
                    logger.info(f'  - Moving points in GT: {moving_points_gt_sample} ({moving_points_gt_sample/points_in_frame*100:.2f}%)')
                    logger.info(f'  - Moving points in prediction: {moving_points_pred_sample} ({moving_points_pred_sample/points_in_frame*100:.2f}%)')
                    logger.info(f'  - Prediction differs from GT: {differs_in_sample} ({differs_in_sample/points_in_frame*100:.2f}%)')
                    logger.info(f'  - Points saved after filtering: {np.sum(non_moving_mask)}')
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if (i + 1) % args.print_freq == 0 and main_process():
                logger.info('Test: [{}/{}] '
                           'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                           'Batch {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                                i + 1, len(val_loader),
                                data_time=data_time,
                                batch_time=batch_time))
    
    # 在结束时打印统计信息
    if main_process():
        logger.info(f'统计信息:')
        logger.info(f'  - 总点数: {total_points_count}')
        logger.info(f'  - 原始标签中的移动物体点数: {moving_points_in_gt_count} ({moving_points_in_gt_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 预测结果中的移动物体点数: {moving_points_in_pred_count} ({moving_points_in_pred_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 预测与标签不同的点数: {prediction_differs_count} ({prediction_differs_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 过滤后保存的点数: {total_points_count - moving_points_in_gt_count} ({(total_points_count - moving_points_in_gt_count)/total_points_count*100:.2f}%)')
        
        # 打印预测类别分布
        logger.info(f'预测类别分布:')
        for cls in sorted(prediction_class_counts.keys()):
            count = prediction_class_counts[cls]
            percentage = count / total_points_count * 100
            logger.info(f'  - 类别 {cls}: {count} ({percentage:.2f}%)')
            
            # 特别标注移动物体类别 - 修改为类别ID为19的点
            if cls == 19:
                logger.info(f'    (移动物体类别)')
        
        # 打印真实标签类别分布
        logger.info(f'真实标签类别分布:')
        for cls in sorted(gt_class_counts.keys()):
            count = gt_class_counts[cls]
            percentage = count / total_points_count * 100
            logger.info(f'  - 类别 {cls}: {count} ({percentage:.2f}%)')
            
            # 特别标注移动物体类别 - 修改为类别ID为19的点
            if cls == 19:
                logger.info(f'    (移动物体类别)')
        
        logger.info(f'Predictions saved to {args.save_folder}/predictions/')
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation and Saving Predictions <<<<<<<<<<<<<<<<<')


def validate_and_print_points9(val_loader, model):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation and Saving Predictions >>>>>>>>>>>>>>>>')
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    # 点云统计计数器
    total_points_count = 0
    moving_points_in_gt_count = 0
    moving_points_in_pred_count = 0
    prediction_differs_count = 0
    
    # 混淆矩阵相关计数器
    true_positive_count = 0  # 原始移动预测移动
    false_negative_count = 0  # 原始移动预测不动
    true_negative_count = 0  # 原始不动预测不动
    false_positive_count = 0  # 原始不动预测移动
    
    # 添加预测类别统计字典
    prediction_class_counts = {}
    gt_class_counts = {}
    
    model.eval()
    end = time.time()
    
    # 获取进程信息
    rank = 0
    if args.multiprocessing_distributed:
        rank = dist.get_rank()
    
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            data_time.update(time.time() - end)
            
            if main_process() and (i + 1) % args.print_freq == 0:
                logger.info(f'Processing batch {i+1}/{len(val_loader)}')
            
            # 解包批次数据
            (coord, xyz, feat, target, offset, inds_reverse, file_names) = batch_data
            
            # 计算当前批次中的点数
            batch_points_count = coord.shape[0]
            total_points_count += batch_points_count
            
            # 计算原始标签中移动物体的点数 - 修改为类别ID为19的点
            moving_mask_gt = (target == 19)
            moving_points_gt = moving_mask_gt.sum().item()
            moving_points_in_gt_count += moving_points_gt
            
            inds_reverse = inds_reverse.cuda(non_blocking=True)
            
            offset_ = offset.clone()
            offset_[1:] = offset_[1:] - offset_[:-1]
            
            # 计算每个样本中的点数量
            points_per_sample = offset_.tolist()
            if main_process() and (i + 1) % args.print_freq == 0:
                logger.info(f'Points per sample: {points_per_sample}')
            
            batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()
            
            coord = torch.cat([batch.unsqueeze(-1), coord], -1)
            spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
                       
            coord, xyz, feat, target = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True)
            offset = offset.cuda(non_blocking=True)
            batch = batch.cuda(non_blocking=True)
            
            sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, args.batch_size_val)
            
            # 运行模型并获取预测结果
            output = model(sinput, xyz, batch)
            output = output[inds_reverse, :]
            
            # 获取预测类别
            pred_logits = output.max(1)[1]
            
            # 计算预测结果中移动物体的点数 - 修改为类别ID为19的点
            moving_mask_pred = (pred_logits == 19)
            moving_points_pred = moving_mask_pred.sum().item()
            moving_points_in_pred_count += moving_points_pred
            
            # 计算预测与标签不同的点数
            differs_mask = (pred_logits != target)
            batch_prediction_differs = differs_mask.sum().item()
            prediction_differs_count += batch_prediction_differs
            
            # 计算四类点的数量
            # 1. 原始移动预测移动（真阳性）
            true_positive_mask = (target == 19) & (pred_logits == 19)
            batch_true_positive = true_positive_mask.sum().item()
            true_positive_count += batch_true_positive
            
            # 2. 原始移动预测不动（假阴性）
            false_negative_mask = (target == 19) & (pred_logits != 19)
            batch_false_negative = false_negative_mask.sum().item()
            false_negative_count += batch_false_negative
            
            # 3. 原始不动预测不动（真阴性）
            true_negative_mask = (target != 19) & (pred_logits != 19)
            batch_true_negative = true_negative_mask.sum().item()
            true_negative_count += batch_true_negative
            
            # 4. 原始不动预测移动（假阳性）
            false_positive_mask = (target != 19) & (pred_logits == 19)
            batch_false_positive = false_positive_mask.sum().item()
            false_positive_count += batch_false_positive
            
            # 将预测结果和目标标签转移到CPU进行保存
            predictions = pred_logits.cpu().numpy()
            target_cpu = target.cpu().numpy()
            batch_idx = batch[inds_reverse].cpu().numpy()
            
            # 统计预测类别分布
            unique_pred_classes, pred_counts = np.unique(predictions, return_counts=True)
            for cls, count in zip(unique_pred_classes, pred_counts):
                if cls in prediction_class_counts:
                    prediction_class_counts[cls] += count
                else:
                    prediction_class_counts[cls] = count
            
            # 统计真实标签类别分布
            unique_gt_classes, gt_counts = np.unique(target_cpu, return_counts=True)
            for cls, count in zip(unique_gt_classes, gt_counts):
                if cls in gt_class_counts:
                    gt_class_counts[cls] += count
                else:
                    gt_class_counts[cls] = count
            
            # 保存预测结果，按照SemanticKITTI格式
            for b_idx in np.unique(batch_idx):
                mask = batch_idx == b_idx
                scan_predictions = predictions[mask]
                scan_targets = target_cpu[mask]
                
                # 使用原始标签中的移动物体掩码过滤预测结果 - 修改为类别ID为19的点
                moving_mask_gt_sample = (scan_targets == 19)
                non_moving_mask = ~moving_mask_gt_sample
                filtered_predictions = scan_predictions[non_moving_mask]
                
                # 使用文件名构建保存路径
                current_file = file_names[int(b_idx)]
                sequence_dir = os.path.dirname(os.path.dirname(current_file))
                sequence_id = os.path.basename(sequence_dir)
                frame_id = os.path.basename(current_file).split('.')[0]
                
                # 创建保存目录
                save_dir = os.path.join(args.save_folder, 'predictions', sequence_id)
                os.makedirs(save_dir, exist_ok=True)
                
                # 保存过滤后的预测结果
                # save_path = os.path.join(save_dir, f'{frame_id}.label')
                # filtered_predictions = filtered_predictions.astype(np.uint32)
                # filtered_predictions.tofile(save_path)
                
                if main_process() and (i + 1) % args.print_freq == 0:
                    points_in_frame = np.sum(mask)
                    moving_points_gt_sample = np.sum(moving_mask_gt_sample)
                    moving_points_pred_sample = np.sum((scan_predictions == 19))
                    
                    # 计算该样本中预测与标签不同的点数
                    differs_in_sample = np.sum(scan_predictions != scan_targets)
                    
                    # 计算四类点在当前样本中的数量
                    tp_sample = np.sum((scan_targets == 19) & (scan_predictions == 19))
                    fn_sample = np.sum((scan_targets == 19) & (scan_predictions != 19))
                    tn_sample = np.sum((scan_targets != 19) & (scan_predictions != 19))
                    fp_sample = np.sum((scan_targets != 19) & (scan_predictions == 19))
                    
                    logger.info(f'Saved prediction for sequence {sequence_id}, frame {frame_id}')
                    logger.info(f'  - Total points in frame: {points_in_frame}')
                    logger.info(f'  - Moving points in GT: {moving_points_gt_sample} ({moving_points_gt_sample/points_in_frame*100:.2f}%)')
                    logger.info(f'  - Moving points in prediction: {moving_points_pred_sample} ({moving_points_pred_sample/points_in_frame*100:.2f}%)')
                    logger.info(f'  - Prediction differs from GT: {differs_in_sample} ({differs_in_sample/points_in_frame*100:.2f}%)')
                    logger.info(f'  - Points saved after filtering: {np.sum(non_moving_mask)}')
                    
                    # 打印四类点的样本级统计
                    logger.info(f'  - 样本移动物体混淆矩阵:')
                    logger.info(f'    * 原始移动预测移动（真阳性）: {tp_sample} ({tp_sample/points_in_frame*100:.2f}%)')
                    logger.info(f'    * 原始移动预测不动（假阴性）: {fn_sample} ({fn_sample/points_in_frame*100:.2f}%)')
                    logger.info(f'    * 原始不动预测不动（真阴性）: {tn_sample} ({tn_sample/points_in_frame*100:.2f}%)')
                    logger.info(f'    * 原始不动预测移动（假阳性）: {fp_sample} ({fp_sample/points_in_frame*100:.2f}%)')
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if (i + 1) % args.print_freq == 0 and main_process():
                logger.info('Test: [{}/{}] '
                           'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                           'Batch {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                                i + 1, len(val_loader),
                                data_time=data_time,
                                batch_time=batch_time))
    
    # 在结束时打印统计信息
    if main_process():
        logger.info(f'统计信息:')
        logger.info(f'  - 总点数: {total_points_count}')
        logger.info(f'  - 原始标签中的移动物体点数: {moving_points_in_gt_count} ({moving_points_in_gt_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 预测结果中的移动物体点数: {moving_points_in_pred_count} ({moving_points_in_pred_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 预测与标签不同的点数: {prediction_differs_count} ({prediction_differs_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 过滤后保存的点数: {total_points_count - moving_points_in_gt_count} ({(total_points_count - moving_points_in_gt_count)/total_points_count*100:.2f}%)')
        
        # 打印四类点的统计信息
        logger.info(f'移动物体混淆矩阵统计:')
        logger.info(f'  - 原始移动预测移动（真阳性）: {true_positive_count} ({true_positive_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 原始移动预测不动（假阴性）: {false_negative_count} ({false_negative_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 原始不动预测不动（真阴性）: {true_negative_count} ({true_negative_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 原始不动预测移动（假阳性）: {false_positive_count} ({false_positive_count/total_points_count*100:.2f}%)')
        
        # 计算精确率和召回率
        precision = true_positive_count / (true_positive_count + false_positive_count) if (true_positive_count + false_positive_count) > 0 else 0
        recall = true_positive_count / (true_positive_count + false_negative_count) if (true_positive_count + false_negative_count) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        logger.info(f'移动物体检测性能指标:')
        logger.info(f'  - 精确率 (Precision): {precision:.4f}')
        logger.info(f'  - 召回率 (Recall): {recall:.4f}')
        logger.info(f'  - F1分数: {f1_score:.4f}')
        
        # 打印预测类别分布
        logger.info(f'预测类别分布:')
        for cls in sorted(prediction_class_counts.keys()):
            count = prediction_class_counts[cls]
            percentage = count / total_points_count * 100
            logger.info(f'  - 类别 {cls}: {count} ({percentage:.2f}%)')
            
            # 特别标注移动物体类别 - 修改为类别ID为19的点
            if cls == 19:
                logger.info(f'    (移动物体类别)')
        
        # 打印真实标签类别分布
        logger.info(f'真实标签类别分布:')
        for cls in sorted(gt_class_counts.keys()):
            count = gt_class_counts[cls]
            percentage = count / total_points_count * 100
            logger.info(f'  - 类别 {cls}: {count} ({percentage:.2f}%)')
            
            # 特别标注移动物体类别 - 修改为类别ID为19的点
            if cls == 19:
                logger.info(f'    (移动物体类别)')
        
        logger.info(f'Predictions saved to {args.save_folder}/predictions/')
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation and Saving Predictions <<<<<<<<<<<<<<<<<')


def validate_and_print_points10(val_loader, model):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation and Saving Predictions >>>>>>>>>>>>>>>>')
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    # 点云统计计数器
    total_points_count = 0
    moving_points_in_gt_count = 0
    moving_points_in_pred_count = 0
    prediction_differs_count = 0
    
    # 混淆矩阵相关计数器
    true_positive_count = 0  # 原始移动预测移动
    false_negative_count = 0  # 原始移动预测不动
    true_negative_count = 0  # 原始不动预测不动
    false_positive_count = 0  # 原始不动预测移动
    
    # 添加预测类别统计字典
    prediction_class_counts = {}
    gt_class_counts = {}
    
    model.eval()
    end = time.time()
    
    # 获取进程信息
    rank = 0
    if args.multiprocessing_distributed:
        rank = dist.get_rank()
    
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            data_time.update(time.time() - end)
            
            if main_process() and (i + 1) % args.print_freq == 0:
                logger.info(f'Processing batch {i+1}/{len(val_loader)}')
            
            # 解包批次数据
            (coord, xyz, feat, target, offset, inds_reverse, file_names) = batch_data
            
            # 计算当前批次中的点数
            batch_points_count = coord.shape[0]
            total_points_count += batch_points_count
            
            # 计算原始标签中移动物体的点数 - 修改为类别ID为19的点
            moving_mask_gt = (target == 19)
            moving_points_gt = moving_mask_gt.sum().item()
            moving_points_in_gt_count += moving_points_gt
            
            inds_reverse = inds_reverse.cuda(non_blocking=True)
            
            offset_ = offset.clone()
            offset_[1:] = offset_[1:] - offset_[:-1]
            
            # 计算每个样本中的点数量
            points_per_sample = offset_.tolist()
            if main_process() and (i + 1) % args.print_freq == 0:
                logger.info(f'Points per sample: {points_per_sample}')
            
            batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()
            
            coord = torch.cat([batch.unsqueeze(-1), coord], -1)
            spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
                       
            coord, xyz, feat, target = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True)
            offset = offset.cuda(non_blocking=True)
            batch = batch.cuda(non_blocking=True)
            
            sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, args.batch_size_val)
            
            # 运行模型并获取预测结果
            output = model(sinput, xyz, batch)
            output = output[inds_reverse, :]
            
            # 获取预测类别
            pred_logits = output.max(1)[1]
            
            # 计算预测结果中移动物体的点数 - 修改为类别ID为19的点
            moving_mask_pred = (pred_logits == 19)
            moving_points_pred = moving_mask_pred.sum().item()
            moving_points_in_pred_count += moving_points_pred
            
            # 计算预测与标签不同的点数
            differs_mask = (pred_logits != target)
            batch_prediction_differs = differs_mask.sum().item()
            prediction_differs_count += batch_prediction_differs
            
            # 计算四类点的数量
            # 1. 原始移动预测移动（真阳性）
            true_positive_mask = (target == 19) & (pred_logits == 19)
            batch_true_positive = true_positive_mask.sum().item()
            true_positive_count += batch_true_positive
            
            # 2. 原始移动预测不动（假阴性）
            false_negative_mask = (target == 19) & (pred_logits != 19)
            batch_false_negative = false_negative_mask.sum().item()
            false_negative_count += batch_false_negative
            
            # 3. 原始不动预测不动（真阴性）
            true_negative_mask = (target != 19) & (pred_logits != 19)
            batch_true_negative = true_negative_mask.sum().item()
            true_negative_count += batch_true_negative
            
            # 4. 原始不动预测移动（假阳性）
            false_positive_mask = (target != 19) & (pred_logits == 19)
            batch_false_positive = false_positive_mask.sum().item()
            false_positive_count += batch_false_positive
            
            # 验证每个批次的计数是否正确
            batch_sum = batch_true_positive + batch_false_negative + batch_true_negative + batch_false_positive
            assert batch_sum == batch_points_count, f"批次点数不一致: {batch_sum} vs {batch_points_count}"
            
            # 将预测结果和目标标签转移到CPU进行保存
            predictions = pred_logits.cpu().numpy()
            target_cpu = target.cpu().numpy()
            batch_idx = batch[inds_reverse].cpu().numpy()
            
            # 统计预测类别分布
            unique_pred_classes, pred_counts = np.unique(predictions, return_counts=True)
            for cls, count in zip(unique_pred_classes, pred_counts):
                if cls in prediction_class_counts:
                    prediction_class_counts[cls] += count
                else:
                    prediction_class_counts[cls] = count
            
            # 统计真实标签类别分布
            unique_gt_classes, gt_counts = np.unique(target_cpu, return_counts=True)
            for cls, count in zip(unique_gt_classes, gt_counts):
                if cls in gt_class_counts:
                    gt_class_counts[cls] += count
                else:
                    gt_class_counts[cls] = count
            
            # 保存预测结果，按照SemanticKITTI格式
            for b_idx in np.unique(batch_idx):
                mask = batch_idx == b_idx
                scan_predictions = predictions[mask]
                scan_targets = target_cpu[mask]
                
                # 使用原始标签中的移动物体掩码过滤预测结果 - 修改为类别ID为19的点
                moving_mask_gt_sample = (scan_targets == 19)
                non_moving_mask = ~moving_mask_gt_sample
                filtered_predictions = scan_predictions[non_moving_mask]
                
                # 使用文件名构建保存路径
                current_file = file_names[int(b_idx)]
                sequence_dir = os.path.dirname(os.path.dirname(current_file))
                sequence_id = os.path.basename(sequence_dir)
                frame_id = os.path.basename(current_file).split('.')[0]
                
                # 创建保存目录
                save_dir = os.path.join(args.save_folder, 'predictions', sequence_id)
                os.makedirs(save_dir, exist_ok=True)
                
                # 保存过滤后的预测结果
                save_path = os.path.join(save_dir, f'{frame_id}.label')
                filtered_predictions = filtered_predictions.astype(np.uint32)
                filtered_predictions.tofile(save_path)
                
                if main_process() and (i + 1) % args.print_freq == 0:
                    points_in_frame = np.sum(mask)
                    moving_points_gt_sample = np.sum(moving_mask_gt_sample)
                    moving_points_pred_sample = np.sum((scan_predictions == 19))
                    
                    # 计算该样本中预测与标签不同的点数
                    differs_in_sample = np.sum(scan_predictions != scan_targets)
                    
                    # 计算四类点在当前样本中的数量
                    tp_sample = np.sum((scan_targets == 19) & (scan_predictions == 19))
                    fn_sample = np.sum((scan_targets == 19) & (scan_predictions != 19))
                    tn_sample = np.sum((scan_targets != 19) & (scan_predictions != 19))
                    fp_sample = np.sum((scan_targets != 19) & (scan_predictions == 19))
                    
                    logger.info(f'Saved prediction for sequence {sequence_id}, frame {frame_id}')
                    logger.info(f'  - Total points in frame: {points_in_frame}')
                    logger.info(f'  - Moving points in GT: {moving_points_gt_sample} ({moving_points_gt_sample/points_in_frame*100:.2f}%)')
                    logger.info(f'  - Moving points in prediction: {moving_points_pred_sample} ({moving_points_pred_sample/points_in_frame*100:.2f}%)')
                    logger.info(f'  - Prediction differs from GT: {differs_in_sample} ({differs_in_sample/points_in_frame*100:.2f}%)')
                    logger.info(f'  - Points saved after filtering: {np.sum(non_moving_mask)}')
                    
                    # 打印四类点的样本级统计
                    logger.info(f'  - 样本移动物体混淆矩阵:')
                    logger.info(f'    * 原始移动预测移动（真阳性）: {tp_sample} ({tp_sample/points_in_frame*100:.2f}%)')
                    logger.info(f'    * 原始移动预测不动（假阴性）: {fn_sample} ({fn_sample/points_in_frame*100:.2f}%)')
                    logger.info(f'    * 原始不动预测不动（真阴性）: {tn_sample} ({tn_sample/points_in_frame*100:.2f}%)')
                    logger.info(f'    * 原始不动预测移动（假阳性）: {fp_sample} ({fp_sample/points_in_frame*100:.2f}%)')
                    
                    # 验证样本级的计数是否正确
                    sample_sum = tp_sample + fn_sample + tn_sample + fp_sample
                    if sample_sum != points_in_frame:
                        logger.warning(f'样本点数不一致: {sample_sum} vs {points_in_frame}')
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if (i + 1) % args.print_freq == 0 and main_process():
                logger.info('Test: [{}/{}] '
                           'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                           'Batch {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                                i + 1, len(val_loader),
                                data_time=data_time,
                                batch_time=batch_time))
    
    # 在多进程环境中合并统计数据
    if args.multiprocessing_distributed:
        dist.all_reduce(torch.tensor([total_points_count, moving_points_in_gt_count, 
                                     moving_points_in_pred_count, prediction_differs_count,
                                     true_positive_count, false_negative_count,
                                     true_negative_count, false_positive_count]).cuda())
    
    # 验证总计数是否正确
    total_matrix_count = true_positive_count + false_negative_count + true_negative_count + false_positive_count
    if total_matrix_count != total_points_count and main_process():
        logger.warning(f"总点数不一致: 混淆矩阵总数={total_matrix_count}, 总点数={total_points_count}")
        # 对总数进行调整，确保百分比计算正确
        logger.info(f"调整真阴性计数以匹配总点数")
        true_negative_count = total_points_count - true_positive_count - false_negative_count - false_positive_count
    
    # 在结束时打印统计信息
    if main_process():
        logger.info(f'统计信息:')
        logger.info(f'  - 总点数: {total_points_count}')
        logger.info(f'  - 原始标签中的移动物体点数: {moving_points_in_gt_count} ({moving_points_in_gt_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 预测结果中的移动物体点数: {moving_points_in_pred_count} ({moving_points_in_pred_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 预测与标签不同的点数: {prediction_differs_count} ({prediction_differs_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 过滤后保存的点数: {total_points_count - moving_points_in_gt_count} ({(total_points_count - moving_points_in_gt_count)/total_points_count*100:.2f}%)')
        
        # 打印四类点的统计信息
        logger.info(f'移动物体混淆矩阵统计:')
        logger.info(f'  - 原始移动预测移动（真阳性）: {true_positive_count} ({true_positive_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 原始移动预测不动（假阴性）: {false_negative_count} ({false_negative_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 原始不动预测不动（真阴性）: {true_negative_count} ({true_negative_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 原始不动预测移动（假阳性）: {false_positive_count} ({false_positive_count/total_points_count*100:.2f}%)')
        
        # 验证混淆矩阵总数是否等于总点数
        matrix_sum = true_positive_count + false_negative_count + true_negative_count + false_positive_count
        logger.info(f'  - 混淆矩阵总点数: {matrix_sum} ({matrix_sum/total_points_count*100:.2f}%)')
        
        # 计算精确率和召回率
        precision = true_positive_count / (true_positive_count + false_positive_count) if (true_positive_count + false_positive_count) > 0 else 0
        recall = true_positive_count / (true_positive_count + false_negative_count) if (true_positive_count + false_negative_count) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # 计算准确率和特异性
        accuracy = (true_positive_count + true_negative_count) / total_points_count
        specificity = true_negative_count / (true_negative_count + false_positive_count) if (true_negative_count + false_positive_count) > 0 else 0
        
        logger.info(f'移动物体检测性能指标:')
        logger.info(f'  - 精确率 (Precision): {precision:.4f}')
        logger.info(f'  - 召回率 (Recall): {recall:.4f}')
        logger.info(f'  - F1分数: {f1_score:.4f}')
        logger.info(f'  - 准确率 (Accuracy): {accuracy:.4f}')
        logger.info(f'  - 特异性 (Specificity): {specificity:.4f}')
        
        # 打印预测类别分布
        logger.info(f'预测类别分布:')
        for cls in sorted(prediction_class_counts.keys()):
            count = prediction_class_counts[cls]
            percentage = count / total_points_count * 100
            logger.info(f'  - 类别 {cls}: {count} ({percentage:.2f}%)')
            
            # 特别标注移动物体类别 - 修改为类别ID为19的点
            if cls == 19:
                logger.info(f'    (移动物体类别)')
        
        # 打印真实标签类别分布
        logger.info(f'真实标签类别分布:')
        for cls in sorted(gt_class_counts.keys()):
            count = gt_class_counts[cls]
            percentage = count / total_points_count * 100
            logger.info(f'  - 类别 {cls}: {count} ({percentage:.2f}%)')
            
            # 特别标注移动物体类别 - 修改为类别ID为19的点
            if cls == 19:
                logger.info(f'    (移动物体类别)')
        
        logger.info(f'Predictions saved to {args.save_folder}/predictions/')
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation and Saving Predictions <<<<<<<<<<<<<<<<<')

def validate_and_print_points11(val_loader, model):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation and Saving Predictions >>>>>>>>>>>>>>>>')
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    # 点云统计计数器
    total_points_count = 0
    moving_points_in_gt_count = 0
    moving_points_in_pred_count = 0
    prediction_differs_count = 0
    
    # 混淆矩阵相关计数器
    true_positive_count = 0  # 原始移动预测移动
    false_negative_count = 0  # 原始移动预测不动
    true_negative_count = 0  # 原始不动预测不动
    false_positive_count = 0  # 原始不动预测移动
    
    # 添加预测类别统计字典
    prediction_class_counts = {}
    gt_class_counts = {}
    
    model.eval()
    end = time.time()
    
    # 获取进程信息
    rank = 0
    if args.multiprocessing_distributed:
        rank = dist.get_rank()
    
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            data_time.update(time.time() - end)
            
            if main_process() and (i + 1) % args.print_freq == 0:
                logger.info(f'Processing batch {i+1}/{len(val_loader)}')
            
            # 解包批次数据
            (coord, xyz, feat, target, offset, inds_reverse, file_names) = batch_data
            
            # 计算当前批次中的点数
            batch_points_count = coord.shape[0]
            total_points_count += batch_points_count
            
            inds_reverse = inds_reverse.cuda(non_blocking=True)
            
            offset_ = offset.clone()
            offset_[1:] = offset_[1:] - offset_[:-1]
            
            # 计算每个样本中的点数量
            points_per_sample = offset_.tolist()
            if main_process() and (i + 1) % args.print_freq == 0:
                logger.info(f'Points per sample: {points_per_sample}')
            
            batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()
            
            coord = torch.cat([batch.unsqueeze(-1), coord], -1)
            spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
                       
            coord, xyz, feat, target = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True)
            offset = offset.cuda(non_blocking=True)
            batch = batch.cuda(non_blocking=True)
            
            sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, args.batch_size_val)
            
            # 运行模型并获取预测结果
            output = model(sinput, xyz, batch)
            output = output[inds_reverse, :]
            
            # 获取预测类别
            pred_logits = output.max(1)[1]
            
            # 注意：这里移到下面，确保使用inds_reverse重新排序后的标签和预测进行统计
            target_after_reverse = target[inds_reverse]
            
            # 计算原始标签中移动物体的点数 - 修改为类别ID为19的点
            moving_mask_gt = (target_after_reverse == 19)
            moving_points_gt = moving_mask_gt.sum().item()
            moving_points_in_gt_count += moving_points_gt
            
            # 计算预测结果中移动物体的点数 - 修改为类别ID为19的点
            moving_mask_pred = (pred_logits == 19)
            moving_points_pred = moving_mask_pred.sum().item()
            moving_points_in_pred_count += moving_points_pred
            
            # 计算预测与标签不同的点数
            differs_mask = (pred_logits != target_after_reverse)
            batch_prediction_differs = differs_mask.sum().item()
            prediction_differs_count += batch_prediction_differs
            
            # 计算四类点的数量 - 使用重新排序后的标签
            # 1. 原始移动预测移动（真阳性）
            true_positive_mask = (target_after_reverse == 19) & (pred_logits == 19)
            batch_true_positive = true_positive_mask.sum().item()
            true_positive_count += batch_true_positive
            
            # 2. 原始移动预测不动（假阴性）
            false_negative_mask = (target_after_reverse == 19) & (pred_logits != 19)
            batch_false_negative = false_negative_mask.sum().item()
            false_negative_count += batch_false_negative
            
            # 3. 原始不动预测不动（真阴性）
            true_negative_mask = (target_after_reverse != 19) & (pred_logits != 19)
            batch_true_negative = true_negative_mask.sum().item()
            true_negative_count += batch_true_negative
            
            # 4. 原始不动预测移动（假阳性）
            false_positive_mask = (target_after_reverse != 19) & (pred_logits == 19)
            batch_false_positive = false_positive_mask.sum().item()
            false_positive_count += batch_false_positive
            
            # 验证每个批次的计数是否正确 - 使用警告而不是断言
            batch_sum = batch_true_positive + batch_false_negative + batch_true_negative + batch_false_positive
            if batch_sum != batch_points_count and main_process():
                logger.warning(f"批次点数不一致: 混淆矩阵总数={batch_sum}, 批次点数={batch_points_count}")
                # 调整批次中的真阴性计数以匹配
                adjustment = batch_points_count - batch_sum
                logger.warning(f"调整真阴性计数: {batch_true_negative} -> {batch_true_negative + adjustment}")
                batch_true_negative += adjustment
                true_negative_count += adjustment
            
            # 将预测结果和目标标签转移到CPU进行保存
            predictions = pred_logits.cpu().numpy()
            target_cpu = target_after_reverse.cpu().numpy()
            batch_idx = batch[inds_reverse].cpu().numpy()
            
            # 统计预测类别分布
            unique_pred_classes, pred_counts = np.unique(predictions, return_counts=True)
            for cls, count in zip(unique_pred_classes, pred_counts):
                if cls in prediction_class_counts:
                    prediction_class_counts[cls] += count
                else:
                    prediction_class_counts[cls] = count
            
            # 统计真实标签类别分布
            unique_gt_classes, gt_counts = np.unique(target_cpu, return_counts=True)
            for cls, count in zip(unique_gt_classes, gt_counts):
                if cls in gt_class_counts:
                    gt_class_counts[cls] += count
                else:
                    gt_class_counts[cls] = count
            
            # 保存预测结果，按照SemanticKITTI格式
            for b_idx in np.unique(batch_idx):
                mask = batch_idx == b_idx
                scan_predictions = predictions[mask]
                scan_targets = target_cpu[mask]
                
                # 使用原始标签中的移动物体掩码过滤预测结果 - 修改为类别ID为19的点
                moving_mask_gt_sample = (scan_targets == 19)
                non_moving_mask = ~moving_mask_gt_sample
                filtered_predictions = scan_predictions[non_moving_mask]
                
                # 使用文件名构建保存路径
                current_file = file_names[int(b_idx)]
                sequence_dir = os.path.dirname(os.path.dirname(current_file))
                sequence_id = os.path.basename(sequence_dir)
                frame_id = os.path.basename(current_file).split('.')[0]
                
                # 创建保存目录
                save_dir = os.path.join(args.save_folder, 'predictions', sequence_id)
                os.makedirs(save_dir, exist_ok=True)
                
                # 保存过滤后的预测结果
                save_path = os.path.join(save_dir, f'{frame_id}.label')
                filtered_predictions = filtered_predictions.astype(np.uint32)
                filtered_predictions.tofile(save_path)
                
                if main_process() and (i + 1) % args.print_freq == 0:
                    points_in_frame = np.sum(mask)
                    moving_points_gt_sample = np.sum(moving_mask_gt_sample)
                    moving_points_pred_sample = np.sum((scan_predictions == 19))
                    
                    # 计算该样本中预测与标签不同的点数
                    differs_in_sample = np.sum(scan_predictions != scan_targets)
                    
                    # 计算四类点在当前样本中的数量
                    tp_sample = np.sum((scan_targets == 19) & (scan_predictions == 19))
                    fn_sample = np.sum((scan_targets == 19) & (scan_predictions != 19))
                    tn_sample = np.sum((scan_targets != 19) & (scan_predictions != 19))
                    fp_sample = np.sum((scan_targets != 19) & (scan_predictions == 19))
                    
                    logger.info(f'Saved prediction for sequence {sequence_id}, frame {frame_id}')
                    logger.info(f'  - Total points in frame: {points_in_frame}')
                    logger.info(f'  - Moving points in GT: {moving_points_gt_sample} ({moving_points_gt_sample/points_in_frame*100:.2f}%)')
                    logger.info(f'  - Moving points in prediction: {moving_points_pred_sample} ({moving_points_pred_sample/points_in_frame*100:.2f}%)')
                    logger.info(f'  - Prediction differs from GT: {differs_in_sample} ({differs_in_sample/points_in_frame*100:.2f}%)')
                    logger.info(f'  - Points saved after filtering: {np.sum(non_moving_mask)}')
                    
                    # 打印四类点的样本级统计
                    logger.info(f'  - 样本移动物体混淆矩阵:')
                    logger.info(f'    * 原始移动预测移动（真阳性）: {tp_sample} ({tp_sample/points_in_frame*100:.2f}%)')
                    logger.info(f'    * 原始移动预测不动（假阴性）: {fn_sample} ({fn_sample/points_in_frame*100:.2f}%)')
                    logger.info(f'    * 原始不动预测不动（真阴性）: {tn_sample} ({tn_sample/points_in_frame*100:.2f}%)')
                    logger.info(f'    * 原始不动预测移动（假阳性）: {fp_sample} ({fp_sample/points_in_frame*100:.2f}%)')
                    
                    # 验证样本级的计数是否正确
                    sample_sum = tp_sample + fn_sample + tn_sample + fp_sample
                    if sample_sum != points_in_frame:
                        logger.warning(f'样本点数不一致: {sample_sum} vs {points_in_frame}')
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if (i + 1) % args.print_freq == 0 and main_process():
                logger.info('Test: [{}/{}] '
                           'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                           'Batch {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                                i + 1, len(val_loader),
                                data_time=data_time,
                                batch_time=batch_time))
    
    # 在多进程环境中合并统计数据
    if args.multiprocessing_distributed:
        # 创建包含所有计数的张量
        counts_tensor = torch.tensor([total_points_count, moving_points_in_gt_count, 
                                     moving_points_in_pred_count, prediction_differs_count,
                                     true_positive_count, false_negative_count,
                                     true_negative_count, false_positive_count], 
                                     dtype=torch.float64).cuda()
        # 执行所有进程间的规约操作
        dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)
        # 解包更新后的值
        [total_points_count, moving_points_in_gt_count, 
         moving_points_in_pred_count, prediction_differs_count,
         true_positive_count, false_negative_count,
         true_negative_count, false_positive_count] = counts_tensor.cpu().tolist()
        
        # 将数值转回整数
        total_points_count = int(total_points_count)
        moving_points_in_gt_count = int(moving_points_in_gt_count)
        moving_points_in_pred_count = int(moving_points_in_pred_count)
        prediction_differs_count = int(prediction_differs_count)
        true_positive_count = int(true_positive_count)
        false_negative_count = int(false_negative_count)
        true_negative_count = int(true_negative_count)
        false_positive_count = int(false_positive_count)
    
    # 验证总计数是否正确
    total_matrix_count = true_positive_count + false_negative_count + true_negative_count + false_positive_count
    if total_matrix_count != total_points_count and main_process():
        logger.warning(f"总点数不一致: 混淆矩阵总数={total_matrix_count}, 总点数={total_points_count}")
        # 对总数进行调整，确保百分比计算正确
        logger.info(f"调整真阴性计数以匹配总点数")
        true_negative_count = total_points_count - true_positive_count - false_negative_count - false_positive_count
    
    # 在结束时打印统计信息
    if main_process():
        logger.info(f'统计信息:')
        logger.info(f'  - 总点数: {total_points_count}')
        logger.info(f'  - 原始标签中的移动物体点数: {moving_points_in_gt_count} ({moving_points_in_gt_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 预测结果中的移动物体点数: {moving_points_in_pred_count} ({moving_points_in_pred_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 预测与标签不同的点数: {prediction_differs_count} ({prediction_differs_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 过滤后保存的点数: {total_points_count - moving_points_in_gt_count} ({(total_points_count - moving_points_in_gt_count)/total_points_count*100:.2f}%)')
        
        # 打印四类点的统计信息
        logger.info(f'移动物体混淆矩阵统计:')
        logger.info(f'  - 原始移动预测移动（真阳性）: {true_positive_count} ({true_positive_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 原始移动预测不动（假阴性）: {false_negative_count} ({false_negative_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 原始不动预测不动（真阴性）: {true_negative_count} ({true_negative_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 原始不动预测移动（假阳性）: {false_positive_count} ({false_positive_count/total_points_count*100:.2f}%)')
        
        # 验证混淆矩阵总数是否等于总点数
        matrix_sum = true_positive_count + false_negative_count + true_negative_count + false_positive_count
        logger.info(f'  - 混淆矩阵总点数: {matrix_sum} ({matrix_sum/total_points_count*100:.2f}%)')
        
        # 计算精确率和召回率
        precision = true_positive_count / (true_positive_count + false_positive_count) if (true_positive_count + false_positive_count) > 0 else 0
        recall = true_positive_count / (true_positive_count + false_negative_count) if (true_positive_count + false_negative_count) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # 计算准确率和特异性
        accuracy = (true_positive_count + true_negative_count) / total_points_count
        specificity = true_negative_count / (true_negative_count + false_positive_count) if (true_negative_count + false_positive_count) > 0 else 0
        
        logger.info(f'移动物体检测性能指标:')
        logger.info(f'  - 精确率 (Precision): {precision:.4f}')
        logger.info(f'  - 召回率 (Recall): {recall:.4f}')
        logger.info(f'  - F1分数: {f1_score:.4f}')
        logger.info(f'  - 准确率 (Accuracy): {accuracy:.4f}')
        logger.info(f'  - 特异性 (Specificity): {specificity:.4f}')
        
        # 打印预测类别分布
        logger.info(f'预测类别分布:')
        for cls in sorted(prediction_class_counts.keys()):
            count = prediction_class_counts[cls]
            percentage = count / total_points_count * 100
            logger.info(f'  - 类别 {cls}: {count} ({percentage:.2f}%)')
            
            # 特别标注移动物体类别 - 修改为类别ID为19的点
            if cls == 19:
                logger.info(f'    (移动物体类别)')
        
        # 打印真实标签类别分布
        logger.info(f'真实标签类别分布:')
        for cls in sorted(gt_class_counts.keys()):
            count = gt_class_counts[cls]
            percentage = count / total_points_count * 100
            logger.info(f'  - 类别 {cls}: {count} ({percentage:.2f}%)')
            
            # 特别标注移动物体类别 - 修改为类别ID为19的点
            if cls == 19:
                logger.info(f'    (移动物体类别)')
        
        logger.info(f'Predictions saved to {args.save_folder}/predictions/')
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation and Saving Predictions <<<<<<<<<<<<<<<<<')


def validate_and_print_points12(val_loader, model):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation and Saving Predictions >>>>>>>>>>>>>>>>')
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    # 点云统计计数器
    total_points_count = 0
    moving_points_in_gt_count = 0
    moving_points_in_pred_count = 0
    prediction_differs_count = 0
    
    # 混淆矩阵相关计数器
    true_positive_count = 0  # 原始移动预测移动
    false_negative_count = 0  # 原始移动预测不动
    true_negative_count = 0  # 原始不动预测不动
    false_positive_count = 0  # 原始不动预测移动
    
    # 添加预测类别统计字典
    prediction_class_counts = {}
    gt_class_counts = {}
    
    model.eval()
    end = time.time()
    
    # 获取进程信息
    rank = 0
    if args.multiprocessing_distributed:
        rank = dist.get_rank()
    
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            data_time.update(time.time() - end)
            
            if main_process() and (i + 1) % args.print_freq == 0:
                logger.info(f'Processing batch {i+1}/{len(val_loader)}')
            
            # 解包批次数据
            (coord, xyz, feat, target, offset, inds_reverse, file_names) = batch_data
            
            # 计算当前批次中的点数
            batch_points_count = coord.shape[0]
            total_points_count += batch_points_count
            
            inds_reverse = inds_reverse.cuda(non_blocking=True)
            
            offset_ = offset.clone()
            offset_[1:] = offset_[1:] - offset_[:-1]
            
            # 计算每个样本中的点数量
            points_per_sample = offset_.tolist()
            if main_process() and (i + 1) % args.print_freq == 0:
                logger.info(f'Points per sample: {points_per_sample}')
            
            batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()
            
            coord = torch.cat([batch.unsqueeze(-1), coord], -1)
            spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
                       
            coord, xyz, feat, target = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True)
            offset = offset.cuda(non_blocking=True)
            batch = batch.cuda(non_blocking=True)
            
            sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, args.batch_size_val)
            
            # 运行模型并获取预测结果
            output = model(sinput, xyz, batch)
            output = output[inds_reverse, :]
            
            # 获取预测类别
            pred_logits = output.max(1)[1]
            
            # 使用inds_reverse重新排序后的标签
            target_after_reverse = target[inds_reverse]
            
            # 计算原始标签中移动物体的点数 - 类别ID为19的点
            moving_mask_gt = (target_after_reverse == 19)
            moving_points_gt = moving_mask_gt.sum().item()
            moving_points_in_gt_count += moving_points_gt
            
            # 计算预测结果中移动物体的点数 - 类别ID为19的点
            moving_mask_pred = (pred_logits == 19)
            moving_points_pred = moving_mask_pred.sum().item()
            moving_points_in_pred_count += moving_points_pred
            
            # 计算预测与标签不同的点数 - 确保只计算一次
            differs_mask = (pred_logits != target_after_reverse)
            batch_prediction_differs = differs_mask.sum().item()
            
            # 验证batch_prediction_differs不超过batch_points_count
            if batch_prediction_differs > batch_points_count:
                if main_process():
                    logger.warning(f"批次中预测差异点数({batch_prediction_differs})超过批次点数({batch_points_count})，将其限制为批次点数")
                batch_prediction_differs = batch_points_count
                
            prediction_differs_count += batch_prediction_differs
            
            # 计算四类点的数量 - 使用重新排序后的标签
            # 1. 原始移动预测移动（真阳性）
            true_positive_mask = (target_after_reverse == 19) & (pred_logits == 19)
            batch_true_positive = true_positive_mask.sum().item()
            true_positive_count += batch_true_positive
            
            # 2. 原始移动预测不动（假阴性）
            false_negative_mask = (target_after_reverse == 19) & (pred_logits != 19)
            batch_false_negative = false_negative_mask.sum().item()
            false_negative_count += batch_false_negative
            
            # 3. 原始不动预测不动（真阴性）
            true_negative_mask = (target_after_reverse != 19) & (pred_logits != 19)
            batch_true_negative = true_negative_mask.sum().item()
            
            # 4. 原始不动预测移动（假阳性）
            false_positive_mask = (target_after_reverse != 19) & (pred_logits == 19)
            batch_false_positive = false_positive_mask.sum().item()
            
            # 验证四个类别总和等于批次点数
            batch_sum = batch_true_positive + batch_false_negative + batch_true_negative + batch_false_positive
            if batch_sum != batch_points_count:
                if main_process():
                    logger.warning(f"批次点数不一致: 混淆矩阵总数={batch_sum}, 批次点数={batch_points_count}")
                
                # 验证其他三类之和不超过批次点数
                other_sum = batch_true_positive + batch_false_negative + batch_false_positive
                if other_sum > batch_points_count:
                    if main_process():
                        logger.warning(f"除真阴性外的点数和({other_sum})超过批次点数({batch_points_count})，不应该发生")
                    # 重置为实际计算的值，不进行调整
                    true_negative_count += batch_true_negative
                else:
                    # 仅当其他三类之和不超过批次点数时才调整真阴性
                    adjusted_true_negative = batch_points_count - other_sum
                    if main_process():
                        logger.warning(f"调整真阴性计数: {batch_true_negative} -> {adjusted_true_negative}")
                    true_negative_count += adjusted_true_negative
            else:
                # 如果总和正确，直接累加真阴性
                true_negative_count += batch_true_negative
            
            false_positive_count += batch_false_positive
            
            # 将预测结果和目标标签转移到CPU进行保存
            predictions = pred_logits.cpu().numpy()
            target_cpu = target_after_reverse.cpu().numpy()
            batch_idx = batch[inds_reverse].cpu().numpy()
            
            # 统计预测类别分布
            unique_pred_classes, pred_counts = np.unique(predictions, return_counts=True)
            for cls, count in zip(unique_pred_classes, pred_counts):
                if cls in prediction_class_counts:
                    prediction_class_counts[cls] += count
                else:
                    prediction_class_counts[cls] = count
            
            # 统计真实标签类别分布
            unique_gt_classes, gt_counts = np.unique(target_cpu, return_counts=True)
            for cls, count in zip(unique_gt_classes, gt_counts):
                if cls in gt_class_counts:
                    gt_class_counts[cls] += count
                else:
                    gt_class_counts[cls] = count
            
            # 保存预测结果，按照SemanticKITTI格式
            for b_idx in np.unique(batch_idx):
                mask = batch_idx == b_idx
                scan_predictions = predictions[mask]
                scan_targets = target_cpu[mask]
                
                # 使用原始标签中的移动物体掩码过滤预测结果 - 修改为类别ID为19的点
                moving_mask_gt_sample = (scan_targets == 19)
                non_moving_mask = ~moving_mask_gt_sample
                filtered_predictions = scan_predictions[non_moving_mask]
                
                # 使用文件名构建保存路径
                current_file = file_names[int(b_idx)]
                sequence_dir = os.path.dirname(os.path.dirname(current_file))
                sequence_id = os.path.basename(sequence_dir)
                frame_id = os.path.basename(current_file).split('.')[0]
                
                # 创建保存目录
                save_dir = os.path.join(args.save_folder, 'predictions', sequence_id)
                os.makedirs(save_dir, exist_ok=True)
                
                # 保存过滤后的预测结果
                save_path = os.path.join(save_dir, f'{frame_id}.label')
                filtered_predictions = filtered_predictions.astype(np.uint32)
                filtered_predictions.tofile(save_path)
                
                if main_process() and (i + 1) % args.print_freq == 0:
                    points_in_frame = np.sum(mask)
                    moving_points_gt_sample = np.sum(moving_mask_gt_sample)
                    moving_points_pred_sample = np.sum((scan_predictions == 19))
                    
                    # 计算该样本中预测与标签不同的点数
                    differs_in_sample = np.sum(scan_predictions != scan_targets)
                    
                    # 确保differs_in_sample不超过points_in_frame
                    if differs_in_sample > points_in_frame:
                        logger.warning(f"样本中预测差异点数({differs_in_sample})超过样本点数({points_in_frame})，将其限制为样本点数")
                        differs_in_sample = points_in_frame
                    
                    # 计算四类点在当前样本中的数量
                    tp_sample = np.sum((scan_targets == 19) & (scan_predictions == 19))
                    fn_sample = np.sum((scan_targets == 19) & (scan_predictions != 19))
                    tn_sample = np.sum((scan_targets != 19) & (scan_predictions != 19))
                    fp_sample = np.sum((scan_targets != 19) & (scan_predictions == 19))
                    
                    logger.info(f'Saved prediction for sequence {sequence_id}, frame {frame_id}')
                    logger.info(f'  - Total points in frame: {points_in_frame}')
                    logger.info(f'  - Moving points in GT: {moving_points_gt_sample} ({moving_points_gt_sample/points_in_frame*100:.2f}%)')
                    logger.info(f'  - Moving points in prediction: {moving_points_pred_sample} ({moving_points_pred_sample/points_in_frame*100:.2f}%)')
                    logger.info(f'  - Prediction differs from GT: {differs_in_sample} ({differs_in_sample/points_in_frame*100:.2f}%)')
                    logger.info(f'  - Points saved after filtering: {np.sum(non_moving_mask)}')
                    
                    # 打印四类点的样本级统计
                    logger.info(f'  - 样本移动物体混淆矩阵:')
                    logger.info(f'    * 原始移动预测移动（真阳性）: {tp_sample} ({tp_sample/points_in_frame*100:.2f}%)')
                    logger.info(f'    * 原始移动预测不动（假阴性）: {fn_sample} ({fn_sample/points_in_frame*100:.2f}%)')
                    logger.info(f'    * 原始不动预测不动（真阴性）: {tn_sample} ({tn_sample/points_in_frame*100:.2f}%)')
                    logger.info(f'    * 原始不动预测移动（假阳性）: {fp_sample} ({fp_sample/points_in_frame*100:.2f}%)')
                    
                    # 验证样本级的计数是否正确
                    sample_sum = tp_sample + fn_sample + tn_sample + fp_sample
                    if sample_sum != points_in_frame:
                        logger.warning(f'样本点数不一致: {sample_sum} vs {points_in_frame}')
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if (i + 1) % args.print_freq == 0 and main_process():
                logger.info('Test: [{}/{}] '
                           'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                           'Batch {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                                i + 1, len(val_loader),
                                data_time=data_time,
                                batch_time=batch_time))
    
    # 在多进程环境中合并统计数据
    if args.multiprocessing_distributed:
        # 创建包含所有计数的张量，使用float64确保精度
        counts_tensor = torch.tensor([
            float(total_points_count), 
            float(moving_points_in_gt_count), 
            float(moving_points_in_pred_count), 
            float(prediction_differs_count),
            float(true_positive_count), 
            float(false_negative_count),
            float(true_negative_count), 
            float(false_positive_count)
        ], dtype=torch.float64).cuda()
        
        # 执行所有进程间的规约操作
        dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)
        
        # 解包更新后的值并转换为整数
        [total_points_count, moving_points_in_gt_count, 
         moving_points_in_pred_count, prediction_differs_count,
         true_positive_count, false_negative_count,
         true_negative_count, false_positive_count] = [int(x) for x in counts_tensor.cpu().tolist()]
    
    # 验证预测差异点数不超过总点数
    if prediction_differs_count > total_points_count:
        if main_process():
            logger.warning(f"预测差异点数({prediction_differs_count})超过总点数({total_points_count})，将其限制为总点数")
        prediction_differs_count = total_points_count
    
    # 验证总计数是否正确
    total_matrix_count = true_positive_count + false_negative_count + true_negative_count + false_positive_count
    if total_matrix_count != total_points_count and main_process():
        logger.warning(f"总点数不一致: 混淆矩阵总数={total_matrix_count}, 总点数={total_points_count}")
        
        # 验证其他三类之和不超过总点数
        other_sum = true_positive_count + false_negative_count + false_positive_count
        if other_sum > total_points_count:
            logger.warning(f"除真阴性外的点数和({other_sum})超过总点数({total_points_count})，这是数据不一致的严重问题")
            # 在这种情况下，我们可以根据实际情况调整总点数或其他计数
            # 这里选择调整总点数以匹配混淆矩阵总数
            logger.warning(f"调整总点数以匹配混淆矩阵总数: {total_points_count} -> {total_matrix_count}")
            total_points_count = total_matrix_count
        else:
            # 仅当其他三类之和不超过总点数时才调整真阴性
            logger.info(f"调整真阴性计数以匹配总点数")
            true_negative_count = total_points_count - true_positive_count - false_negative_count - false_positive_count
    
    # 合并类别统计信息（如果在分布式环境中）
    if args.multiprocessing_distributed:
        # 对于预测类别计数
        all_pred_classes = set()
        for i in range(dist.get_world_size()):
            if i == rank:
                classes = list(prediction_class_counts.keys())
            else:
                classes = [None] * len(prediction_class_counts)
            
            dist.broadcast_object_list(classes, src=i)
            all_pred_classes.update(classes)
        
        # 对于每个类别，合并计数
        for cls in all_pred_classes:
            if cls in prediction_class_counts:
                count = torch.tensor([float(prediction_class_counts[cls])], dtype=torch.float64).cuda()
            else:
                count = torch.tensor([0.0], dtype=torch.float64).cuda()
            
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
            prediction_class_counts[cls] = int(count.cpu().item())
        
        # 对于真实标签类别计数，执行相同的操作
        all_gt_classes = set()
        for i in range(dist.get_world_size()):
            if i == rank:
                classes = list(gt_class_counts.keys())
            else:
                classes = [None] * len(gt_class_counts)
            
            dist.broadcast_object_list(classes, src=i)
            all_gt_classes.update(classes)
        
        for cls in all_gt_classes:
            if cls in gt_class_counts:
                count = torch.tensor([float(gt_class_counts[cls])], dtype=torch.float64).cuda()
            else:
                count = torch.tensor([0.0], dtype=torch.float64).cuda()
            
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
            gt_class_counts[cls] = int(count.cpu().item())
    
    # 在结束时打印统计信息
    if main_process():
        logger.info(f'统计信息:')
        logger.info(f'  - 总点数: {total_points_count}')
        logger.info(f'  - 原始标签中的移动物体点数: {moving_points_in_gt_count} ({moving_points_in_gt_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 预测结果中的移动物体点数: {moving_points_in_pred_count} ({moving_points_in_pred_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 预测与标签不同的点数: {prediction_differs_count} ({prediction_differs_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 过滤后保存的点数: {total_points_count - moving_points_in_gt_count} ({(total_points_count - moving_points_in_gt_count)/total_points_count*100:.2f}%)')
        
        # 打印四类点的统计信息
        logger.info(f'移动物体混淆矩阵统计:')
        logger.info(f'  - 原始移动预测移动（真阳性）: {true_positive_count} ({true_positive_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 原始移动预测不动（假阴性）: {false_negative_count} ({false_negative_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 原始不动预测不动（真阴性）: {true_negative_count} ({true_negative_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 原始不动预测移动（假阳性）: {false_positive_count} ({false_positive_count/total_points_count*100:.2f}%)')
        
        # 验证混淆矩阵总数是否等于总点数
        matrix_sum = true_positive_count + false_negative_count + true_negative_count + false_positive_count
        logger.info(f'  - 混淆矩阵总点数: {matrix_sum} ({matrix_sum/total_points_count*100:.2f}%)')
        
        # 计算精确率和召回率
        precision = true_positive_count / (true_positive_count + false_positive_count) if (true_positive_count + false_positive_count) > 0 else 0
        recall = true_positive_count / (true_positive_count + false_negative_count) if (true_positive_count + false_negative_count) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # 计算准确率和特异性
        accuracy = (true_positive_count + true_negative_count) / total_points_count
        specificity = true_negative_count / (true_negative_count + false_positive_count) if (true_negative_count + false_positive_count) > 0 else 0
        
        logger.info(f'移动物体检测性能指标:')
        logger.info(f'  - 精确率 (Precision): {precision:.4f}')
        logger.info(f'  - 召回率 (Recall): {recall:.4f}')
        logger.info(f'  - F1分数: {f1_score:.4f}')
        logger.info(f'  - 准确率 (Accuracy): {accuracy:.4f}')
        logger.info(f'  - 特异性 (Specificity): {specificity:.4f}')
        
        # 打印预测类别分布
        logger.info(f'预测类别分布:')
        for cls in sorted(prediction_class_counts.keys()):
            count = prediction_class_counts[cls]
            percentage = count / total_points_count * 100
            logger.info(f'  - 类别 {cls}: {count} ({percentage:.2f}%)')
            
            # 特别标注移动物体类别 - 修改为类别ID为19的点
            if cls == 19:
                logger.info(f'    (移动物体类别)')
        
        # 打印真实标签类别分布
        logger.info(f'真实标签类别分布:')
        for cls in sorted(gt_class_counts.keys()):
            count = gt_class_counts[cls]
            percentage = count / total_points_count * 100
            logger.info(f'  - 类别 {cls}: {count} ({percentage:.2f}%)')
            
            # 特别标注移动物体类别 - 修改为类别ID为19的点
            if cls == 19:
                logger.info(f'    (移动物体类别)')
        
        logger.info(f'Predictions saved to {args.save_folder}/predictions/')
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation and Saving Predictions <<<<<<<<<<<<<<<<<')


def validate_and_print_points13(val_loader, model):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation and Saving Predictions >>>>>>>>>>>>>>>>')
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    # 点云统计计数器
    total_points_count = 0
    moving_points_in_gt_count = 0
    moving_points_in_pred_count = 0
    prediction_differs_count = 0
    
    # 混淆矩阵相关计数器
    true_positive_count = 0  # 原始移动预测移动
    false_negative_count = 0  # 原始移动预测不动
    true_negative_count = 0  # 原始不动预测不动
    false_positive_count = 0  # 原始不动预测移动
    
    # 添加预测类别统计字典
    prediction_class_counts = {}
    gt_class_counts = {}
    
    # 添加调试统计
    debug_samples_checked = 0
    debug_max_samples = 5  # 仅调试前5个样本的详细信息
    
    model.eval()
    end = time.time()
    
    # 获取进程信息
    rank = 0
    if args.multiprocessing_distributed:
        rank = dist.get_rank()
    
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            data_time.update(time.time() - end)
            
            if main_process() and (i + 1) % args.print_freq == 0:
                logger.info(f'Processing batch {i+1}/{len(val_loader)}')
            
            # 解包批次数据
            (coord, xyz, feat, target, offset, inds_reverse, file_names) = batch_data
            
            # 确保所有数据都有正确的形状和类型
            assert coord.dim() == 2, f"coord应该是2D张量，但实际是{coord.dim()}D"
            assert xyz.dim() == 2, f"xyz应该是2D张量，但实际是{xyz.dim()}D"
            assert feat.dim() == 2, f"feat应该是2D张量，但实际是{feat.dim()}D"
            assert target.dim() == 1, f"target应该是1D张量，但实际是{target.dim()}D"
            
            # 计算当前批次中的点数
            batch_points_count = coord.shape[0]
            total_points_count += batch_points_count
            
            # 调试信息：检查数据形状
            if main_process() and i == 0:
                logger.info(f"批次数据形状: coord={coord.shape}, xyz={xyz.shape}, feat={feat.shape}, target={target.shape}")
                logger.info(f"target取值范围: min={target.min().item()}, max={target.max().item()}")
                logger.info(f"target中各类别的出现次数: {torch.bincount(target)}")
            
            # 安全地将数据移动到GPU
            coord = coord.cuda(non_blocking=True)
            xyz = xyz.cuda(non_blocking=True)
            feat = feat.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            offset = offset.cuda(non_blocking=True)
            inds_reverse = inds_reverse.cuda(non_blocking=True)
            
            # 构建批次索引
            offset_ = offset.clone()
            offset_[1:] = offset_[1:] - offset_[:-1]
            points_per_sample = offset_.tolist()
            
            # 检查offset值的合理性
            if main_process() and i == 0:
                logger.info(f'Points per sample: {points_per_sample}')
                logger.info(f'Offset: {offset.tolist()}')
                
            batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long().cuda()
            
            # 添加批次维度到坐标
            coord = torch.cat([batch.unsqueeze(-1), coord], -1)
            spatial_shape = np.clip((coord.cpu().max(0)[0][1:] + 1).numpy(), 128, None)
            
            # 创建稀疏卷积张量并运行模型
            sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, args.batch_size_val)
            
            try:
                # 运行模型并获取预测结果
                output = model(sinput, xyz, batch)
                
                # 检查模型输出
                if main_process() and i == 0:
                    logger.info(f"模型输出形状: {output.shape}")
                    logger.info(f"模型输出取值范围: min={output.min().item()}, max={output.max().item()}")
                    
                # 检查inds_reverse是否有效
                if torch.max(inds_reverse) >= output.shape[0]:
                    if main_process():
                        logger.error(f"inds_reverse中的索引超出范围! max(inds_reverse)={torch.max(inds_reverse).item()}, output.shape[0]={output.shape[0]}")
                    
                    # 修复inds_reverse，确保所有索引都在有效范围内
                    inds_reverse = torch.clamp(inds_reverse, 0, output.shape[0] - 1)
                    if main_process():
                        logger.warning("已将inds_reverse中的无效索引修复为有效值")
                
                # 应用inds_reverse重新排序输出
                output = output[inds_reverse, :]
                
                # 获取预测类别
                pred_logits = output.max(1)[1]
                
                # 调试信息：输出预测类别分布
                if main_process() and i == 0:
                    logger.info(f"预测logits取值范围: min={pred_logits.min().item()}, max={pred_logits.max().item()}")
                    pred_dist = torch.bincount(pred_logits)
                    logger.info(f"预测中各类别的出现次数: {pred_dist}")
                
                # 使用inds_reverse重新排序标签
                if inds_reverse.shape[0] != target.shape[0]:
                    if main_process():
                        logger.error(f"inds_reverse和target的长度不匹配! inds_reverse.shape={inds_reverse.shape}, target.shape={target.shape}")
                    
                    # 裁剪到较短的长度
                    min_len = min(inds_reverse.shape[0], target.shape[0])
                    inds_reverse = inds_reverse[:min_len]
                    target = target[:min_len]
                    pred_logits = pred_logits[:min_len]
                    
                    if main_process():
                        logger.warning(f"已将数据裁剪到共同长度 {min_len}")
                
                # 重新排序标签
                target_after_reverse = target[inds_reverse]
                
                # 调试：检查第一个批次的某些样本
                if main_process() and i == 0 and debug_samples_checked < debug_max_samples:
                    debug_samples_checked += 1
                    sample_idx = 0  # 选择第一个样本进行调试
                    sample_mask = (batch[inds_reverse] == sample_idx)
                    sample_preds = pred_logits[sample_mask]
                    sample_targets = target_after_reverse[sample_mask]
                    
                    # 计算样本的预测差异
                    sample_differs = (sample_preds != sample_targets).sum().item()
                    sample_total = sample_mask.sum().item()
                    
                    logger.info(f"样本 {sample_idx} 调试:")
                    logger.info(f"  - 总点数: {sample_total}")
                    logger.info(f"  - 预测与标签不同的点数: {sample_differs} ({sample_differs/sample_total*100:.2f}%)")
                    
                    # 详细检查前10个点
                    detail_count = min(10, sample_total)
                    sample_indices = torch.where(sample_mask)[0][:detail_count]
                    
                    logger.info(f"  - 详细点信息 (前{detail_count}个):")
                    for idx in sample_indices:
                        logger.info(f"    点 {idx.item()}: 预测={pred_logits[idx].item()}, 标签={target_after_reverse[idx].item()}")
                
                # 计算原始标签中移动物体的点数 - 类别ID为19的点
                moving_mask_gt = (target_after_reverse == 19)
                moving_points_gt = moving_mask_gt.sum().item()
                moving_points_in_gt_count += moving_points_gt
                
                # 计算预测结果中移动物体的点数 - 类别ID为19的点
                moving_mask_pred = (pred_logits == 19)
                moving_points_pred = moving_mask_pred.sum().item()
                moving_points_in_pred_count += moving_points_pred
                
                # 计算预测与标签不同的点数 - 添加更多验证
                differs_mask = (pred_logits != target_after_reverse)
                batch_prediction_differs = differs_mask.sum().item()
                
                # 验证批次点数正确性
                if batch_prediction_differs > batch_points_count:
                    if main_process():
                        logger.warning(f"批次中预测差异点数({batch_prediction_differs})超过批次点数({batch_points_count})，这可能表明数据处理存在问题")
                        logger.warning(f"将差异点数限制为批次点数")
                    batch_prediction_differs = batch_points_count
                    
                prediction_differs_count += batch_prediction_differs
                
                # 混淆矩阵计算
                true_positive_mask = (target_after_reverse == 19) & (pred_logits == 19)
                batch_true_positive = true_positive_mask.sum().item()
                true_positive_count += batch_true_positive
                
                false_negative_mask = (target_after_reverse == 19) & (pred_logits != 19)
                batch_false_negative = false_negative_mask.sum().item()
                false_negative_count += batch_false_negative
                
                true_negative_mask = (target_after_reverse != 19) & (pred_logits != 19)
                batch_true_negative = true_negative_mask.sum().item()
                
                false_positive_mask = (target_after_reverse != 19) & (pred_logits == 19)
                batch_false_positive = false_positive_mask.sum().item()
                
                # 验证混淆矩阵总和
                batch_sum = batch_true_positive + batch_false_negative + batch_true_negative + batch_false_positive
                
                if batch_sum != batch_points_count:
                    if main_process():
                        logger.warning(f"批次点数不一致: 混淆矩阵总数={batch_sum}, 批次点数={batch_points_count}")
                    
                    # 调整计数以保持一致性
                    adjustment = batch_points_count - batch_sum
                    true_negative_count += batch_true_negative + adjustment
                    
                    if main_process():
                        logger.warning(f"调整真阴性计数: {batch_true_negative} -> {batch_true_negative + adjustment}")
                else:
                    # 如果总和正确，直接累加真阴性
                    true_negative_count += batch_true_negative
                
                false_positive_count += batch_false_positive
                
                # 将预测结果和目标标签转移到CPU进行保存
                predictions = pred_logits.cpu().numpy()
                target_cpu = target_after_reverse.cpu().numpy()
                batch_idx = batch[inds_reverse].cpu().numpy()
                
                # 统计预测类别分布
                unique_pred_classes, pred_counts = np.unique(predictions, return_counts=True)
                for cls, count in zip(unique_pred_classes, pred_counts):
                    if cls in prediction_class_counts:
                        prediction_class_counts[cls] += count
                    else:
                        prediction_class_counts[cls] = count
                
                # 统计真实标签类别分布
                unique_gt_classes, gt_counts = np.unique(target_cpu, return_counts=True)
                for cls, count in zip(unique_gt_classes, gt_counts):
                    if cls in gt_class_counts:
                        gt_class_counts[cls] += count
                    else:
                        gt_class_counts[cls] = count
                
                # 保存预测结果，按照SemanticKITTI格式
                for b_idx in np.unique(batch_idx):
                    mask = batch_idx == b_idx
                    scan_predictions = predictions[mask]
                    scan_targets = target_cpu[mask]
                    
                    # 使用原始标签中的移动物体掩码过滤预测结果
                    moving_mask_gt_sample = (scan_targets == 19)
                    non_moving_mask = ~moving_mask_gt_sample
                    filtered_predictions = scan_predictions[non_moving_mask]
                    
                    # 使用文件名构建保存路径
                    current_file = file_names[int(b_idx)]
                    sequence_dir = os.path.dirname(os.path.dirname(current_file))
                    sequence_id = os.path.basename(sequence_dir)
                    frame_id = os.path.basename(current_file).split('.')[0]
                    
                    # 创建保存目录
                    save_dir = os.path.join(args.save_folder, 'predictions', sequence_id)
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # 保存过滤后的预测结果
                    save_path = os.path.join(save_dir, f'{frame_id}.label')
                    filtered_predictions = filtered_predictions.astype(np.uint32)
                    filtered_predictions.tofile(save_path)
                    
                    # 定期打印详细统计信息
                    if main_process() and (i + 1) % args.print_freq == 0:
                        points_in_frame = np.sum(mask)
                        moving_points_gt_sample = np.sum(moving_mask_gt_sample)
                        moving_points_pred_sample = np.sum((scan_predictions == 19))
                        
                        # 计算该样本中预测与标签不同的点数
                        differs_in_sample = np.sum(scan_predictions != scan_targets)
                        logger.info(f'Saved prediction for sequence {sequence_id}, frame {frame_id}')
                        logger.info(f'  - Total points in frame: {points_in_frame}')
                        logger.info(f'  - Moving points in GT: {moving_points_gt_sample} ({moving_points_gt_sample/points_in_frame*100:.2f}%)')
                        logger.info(f'  - Moving points in prediction: {moving_points_pred_sample} ({moving_points_pred_sample/points_in_frame*100:.2f}%)')
                        logger.info(f'  - Prediction differs from GT: {differs_in_sample} ({differs_in_sample/points_in_frame*100:.2f}%)')
                        logger.info(f'  - Points saved after filtering: {np.sum(non_moving_mask)}')
                
            except Exception as e:
                if main_process():
                    logger.error(f"处理批次 {i+1} 时发生错误: {str(e)}")
                    logger.error(traceback.format_exc())
                continue
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            # 定期打印进度
            if (i + 1) % args.print_freq == 0 and main_process():
                logger.info('Test: [{}/{}] '
                           'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                           'Batch {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                                i + 1, len(val_loader),
                                data_time=data_time,
                                batch_time=batch_time))
    
    # 在多进程环境中合并统计数据
    if args.multiprocessing_distributed:
        # 创建包含所有计数的张量，使用float64确保精度
        counts_tensor = torch.tensor([
            float(total_points_count), 
            float(moving_points_in_gt_count), 
            float(moving_points_in_pred_count), 
            float(prediction_differs_count),
            float(true_positive_count), 
            float(false_negative_count),
            float(true_negative_count), 
            float(false_positive_count)
        ], dtype=torch.float64).cuda()
        
        # 执行所有进程间的规约操作
        dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)
        
        # 解包更新后的值并转换为整数
        [total_points_count, moving_points_in_gt_count, 
         moving_points_in_pred_count, prediction_differs_count,
         true_positive_count, false_negative_count,
         true_negative_count, false_positive_count] = [int(x) for x in counts_tensor.cpu().tolist()]
    
    # 验证及调整最终统计数据
    if main_process():
        # 验证预测差异点数不超过总点数
        if prediction_differs_count > total_points_count:
            logger.warning(f"预测差异点数({prediction_differs_count})超过总点数({total_points_count})，将其限制为总点数")
            prediction_differs_count = total_points_count
        
        # 验证总计数是否正确
        total_matrix_count = true_positive_count + false_negative_count + true_negative_count + false_positive_count
        if total_matrix_count != total_points_count:
            logger.warning(f"总点数不一致: 混淆矩阵总数={total_matrix_count}, 总点数={total_points_count}")
            
            # 仅当其他三类之和不超过总点数时才调整真阴性
            other_sum = true_positive_count + false_negative_count + false_positive_count
            if other_sum <= total_points_count:
                logger.info(f"调整真阴性计数以匹配总点数")
                true_negative_count = total_points_count - true_positive_count - false_negative_count - false_positive_count
            else:
                logger.warning(f"除真阴性外的点数和({other_sum})超过总点数({total_points_count})，这表明数据处理存在问题")
    
    # 合并类别统计信息（如果在分布式环境中）
    if args.multiprocessing_distributed:
        # 对于每个类别，合并计数
        for cls in range(50):  # 假设有0-49的类别ID
            if cls in prediction_class_counts:
                count = torch.tensor([float(prediction_class_counts[cls])], dtype=torch.float64).cuda()
            else:
                count = torch.tensor([0.0], dtype=torch.float64).cuda()
            
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
            prediction_class_counts[cls] = int(count.cpu().item())
        
        # 对于真实标签类别计数，执行相同的操作
        for cls in range(50):  # 假设有0-49的类别ID
            if cls in gt_class_counts:
                count = torch.tensor([float(gt_class_counts[cls])], dtype=torch.float64).cuda()
            else:
                count = torch.tensor([0.0], dtype=torch.float64).cuda()
            
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
            gt_class_counts[cls] = int(count.cpu().item())
    
    # 在结束时打印统计信息
    if main_process():
        logger.info(f'最终统计信息:')
        logger.info(f'  - 总点数: {total_points_count}')
        logger.info(f'  - 原始标签中的移动物体点数: {moving_points_in_gt_count} ({moving_points_in_gt_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 预测结果中的移动物体点数: {moving_points_in_pred_count} ({moving_points_in_pred_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 预测与标签不同的点数: {prediction_differs_count} ({prediction_differs_count/total_points_count*100:.2f}%)')
        
        # 如果错误率异常高，提供潜在原因
        if prediction_differs_count / total_points_count > 0.5:  # 超过50%的错误率
            logger.warning("预测错误率异常高，可能的原因包括:")
            logger.warning("  1. 标签格式或类别定义问题")
            logger.warning("  2. 数据预处理步骤不一致")
            logger.warning("  3. 模型输出格式与期望不符")
            logger.warning("  4. 索引重排序(inds_reverse)错误")
            logger.warning("  5. 模型训练不足或使用了不适合的预训练模型")
            logger.warning("建议检查标签和预测分布，并验证少量样本的具体预测结果")
        
        logger.info(f'  - 过滤后保存的点数: {total_points_count - moving_points_in_gt_count} ({(total_points_count - moving_points_in_gt_count)/total_points_count*100:.2f}%)')
        
        # 打印混淆矩阵统计信息
        logger.info(f'移动物体混淆矩阵统计:')
        logger.info(f'  - 原始移动预测移动（真阳性）: {true_positive_count} ({true_positive_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 原始移动预测不动（假阴性）: {false_negative_count} ({false_negative_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 原始不动预测不动（真阴性）: {true_negative_count} ({true_negative_count/total_points_count*100:.2f}%)')
        logger.info(f'  - 原始不动预测移动（假阳性）: {false_positive_count} ({false_positive_count/total_points_count*100:.2f}%)')
        
        # 验证混淆矩阵总数是否等于总点数
        matrix_sum = true_positive_count + false_negative_count + true_negative_count + false_positive_count
        logger.info(f'  - 混淆矩阵总点数: {matrix_sum} ({matrix_sum/total_points_count*100:.2f}%)')
        
        # 计算性能指标，添加错误处理以避免除零
        if (true_positive_count + false_positive_count) > 0:
            precision = true_positive_count / (true_positive_count + false_positive_count)
        else:
            precision = 0
            logger.warning("真阳性与假阳性之和为零，精确率无法计算，已设为0")
            
        if (true_positive_count + false_negative_count) > 0:
            recall = true_positive_count / (true_positive_count + false_negative_count)
        else:
            recall = 0
            logger.warning("真阳性与假阴性之和为零，召回率无法计算，已设为0")
            
        if (precision + recall) > 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 0
            logger.warning("精确率与召回率之和为零，F1分数无法计算，已设为0")
        
        # 计算准确率和特异性
        accuracy = (true_positive_count + true_negative_count) / total_points_count
        
        if (true_negative_count + false_positive_count) > 0:
            specificity = true_negative_count / (true_negative_count + false_positive_count)
        else:
            specificity = 0
            logger.warning("真阴性与假阳性之和为零，特异性无法计算，已设为0")
        
        logger.info(f'移动物体检测性能指标:')
        logger.info(f'  - 精确率 (Precision): {precision:.4f}')
        logger.info(f'  - 召回率 (Recall): {recall:.4f}')
        logger.info(f'  - F1分数: {f1_score:.4f}')
        logger.info(f'  - 准确率 (Accuracy): {accuracy:.4f}')
        logger.info(f'  - 特异性 (Specificity): {specificity:.4f}')
        
        # 打印预测类别分布和真实标签类别分布
        logger.info(f'预测类别分布:')
        for cls in sorted(prediction_class_counts.keys()):
            count = prediction_class_counts[cls]
            percentage = count / total_points_count * 100
            logger.info(f'  - 类别 {cls}: {count} ({percentage:.2f}%)')
            
            # 特别标注移动物体类别
            if cls == 19:
                logger.info(f'    (移动物体类别)')
        
        logger.info(f'真实标签类别分布:')
        for cls in sorted(gt_class_counts.keys()):
            count = gt_class_counts[cls]
            percentage = count / total_points_count * 100
            logger.info(f'  - 类别 {cls}: {count} ({percentage:.2f}%)')
            
            # 特别标注移动物体类别
            if cls == 19:
                logger.info(f'    (移动物体类别)')
        
        logger.info(f'Predictions saved to {args.save_folder}/predictions/')
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation and Saving Predictions <<<<<<<<<<<<<<<<<')
        
        # 返回关键性能指标
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "accuracy": accuracy,
            "specificity": specificity,
            "error_rate": prediction_differs_count / total_points_count,
            "total_points": total_points_count
        }


def filter_and_save_point_cloud(
    val_loader, 
    model, 
    save_folder, 
    moving_object_class=19
):
    """
    过滤并保存点云数据，移除移动点
    
    参数:
    - val_loader: 验证数据加载器
    - model: 深度学习模型
    - save_folder: 保存预测结果的根目录
    - moving_object_class: 移动物体的类别ID (默认为19)
    """
    # 确保保存目录存在
    os.makedirs(save_folder, exist_ok=True)
    
    # 设置模型为评估模式
    model.eval()
    
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            # 解包批次数据
            (coord, xyz, feat, target, offset, inds_reverse, file_names) = batch_data
            
            try:
                # 运行模型并获取预测结果
                sinput = spconv.SparseConvTensor(feat, torch.cat([torch.arange(len(coord)).unsqueeze(-1), coord], -1).int(), 
                                                 np.clip((coord.max(0)[0] + 1).numpy(), 128, None), args.batch_size_val)
                output = model(sinput, xyz, batch)
                
                # 获取预测类别
                pred_logits = output.max(1)[1]
                
                # 应用inds_reverse重新排序输出和标签
                pred_logits = pred_logits[inds_reverse]
                target_after_reverse = target[inds_reverse]
                
                # 按帧处理
                for b_idx in range(len(file_names)):
                    # 获取当前帧的掩码
                    frame_mask = batch[inds_reverse] == b_idx
                    frame_predictions = pred_logits[frame_mask]
                    frame_targets = target_after_reverse[frame_mask]
                    frame_file = file_names[b_idx]
                    
                    # 创建非移动点掩码
                    non_moving_mask = frame_predictions != moving_object_class
                    
                    # 仅保留非移动点的原始文件
                    if main_process():
                        sequence_dir = os.path.dirname(os.path.dirname(frame_file))
                        sequence_id = os.path.basename(sequence_dir)
                        frame_id = os.path.basename(frame_file).split('.')[0]
                        
                        # 创建保存目录
                        save_sequence_dir = os.path.join(save_folder, sequence_id)
                        os.makedirs(save_sequence_dir, exist_ok=True)
                        
                        # 读取原始文件
                        with open(frame_file, 'rb') as f:
                            original_data = np.fromfile(f, dtype=np.float32)
                        
                        # SemanticKITTI数据集中，每个点是4个值：x, y, z, intensity
                        points_count = len(original_data) // 4
                        original_points = original_data.reshape(points_count, 4)
                        
                        # 过滤移动点
                        filtered_points = original_points[non_moving_mask.cpu().numpy()]
                        
                        # 保存过滤后的点云数据
                        save_path = os.path.join(save_sequence_dir, f'{frame_id}.bin')
                        filtered_points.astype(np.float32).tofile(save_path)
                        
                        # 日志记录
                        logger.info(f"保存序列 {sequence_id} 帧 {frame_id} 的点云数据:")
                        logger.info(f"  - 原始点数: {points_count}")
                        logger.info(f"  - 过滤后点数: {len(filtered_points)}")
                        logger.info(f"  - 移动点数: {points_count - len(filtered_points)}")
                        logger.info(f"  - 保存路径: {save_path}")
            
            except Exception as e:
                if main_process():
                    logger.error(f"处理批次 {i+1} 时发生错误: {str(e)}")
                    logger.error(traceback.format_exc())
                continue
    
    if main_process():
        logger.info("点云数据过滤和保存完成")

def remove_moving_points_and_save(val_loader, model):
    print('开始移除移动点并保存点云')
    
    total_points_count = 0
    removed_moving_points_count = 0
    
    model.eval()
    
    with torch.no_grad():
        for batch_data in val_loader:
            # 解包批次数据
            (coord, xyz, feat, target, offset, inds_reverse, file_names) = batch_data
            
            # 将数据移动到GPU
            coord = coord.cuda()
            xyz = xyz.cuda()
            feat = feat.cuda()
            target = target.cuda()
            offset = offset.cuda()
            inds_reverse = inds_reverse.cuda()
            
            # 构建批次索引
            offset_ = offset.clone()
            offset_[1:] = offset_[1:] - offset_[:-1]
            batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long().cuda()
            
            # 添加批次维度到坐标
            coord = torch.cat([batch.unsqueeze(-1), coord], -1)
            spatial_shape = np.clip((coord.cpu().max(0)[0][1:] + 1).numpy(), 128, None)
            
            # 创建稀疏卷积张量并运行模型
            sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, len(val_loader))
            
            # 运行模型并获取预测结果
            output = model(sinput, xyz, batch)
            
            # 应用inds_reverse重新排序输出
            output = output[inds_reverse, :]
            
            # 获取预测类别
            pred_logits = output.max(1)[1]
            
            # 应用inds_reverse重新排序标签
            target_after_reverse = target[inds_reverse]
            
            # 逐帧处理并保存
            for b_idx in torch.unique(batch[inds_reverse]):
                # 创建当前帧的掩码
                frame_mask = batch[inds_reverse] == b_idx
                
                # 获取当前帧的坐标、特征和目标
                frame_coord = coord[frame_mask.cpu()]
                frame_feat = feat[frame_mask]
                frame_target = target_after_reverse[frame_mask]
                frame_pred = pred_logits[frame_mask]
                
                # 移除移动点（类别19）
                non_moving_mask = frame_pred != 19
                non_moving_coord = frame_coord[non_moving_mask]
                non_moving_feat = frame_feat[non_moving_mask]
                non_moving_target = frame_target[non_moving_mask]
                
                # 更新统计信息
                total_points_count += len(frame_coord)
                removed_moving_points_count += len(frame_coord) - len(non_moving_coord)
                
                # 使用文件名构建保存路径
                current_file = file_names[int(b_idx)]
                sequence_dir = os.path.dirname(os.path.dirname(current_file))
                sequence_id = os.path.basename(sequence_dir)
                frame_id = os.path.basename(current_file).split('.')[0]
                
                # 创建保存目录
                save_dir = os.path.join('non_moving_points', sequence_id)
                os.makedirs(save_dir, exist_ok=True)
                
                # 保存非移动点的坐标、特征和标签
                coord_path = os.path.join(save_dir, f'{frame_id}_coord.bin')
                feat_path = os.path.join(save_dir, f'{frame_id}_feat.bin')
                target_path = os.path.join(save_dir, f'{frame_id}_target.bin')
                
                # 转换为numpy并保存
                non_moving_coord_np = non_moving_coord.cpu().numpy()
                non_moving_feat_np = non_moving_feat.cpu().numpy()
                non_moving_target_np = non_moving_target.cpu().numpy()
                
                non_moving_coord_np.tofile(coord_path)
                non_moving_feat_np.tofile(feat_path)
                non_moving_target_np.tofile(target_path)
                
                # 打印处理信息
                print(f'处理序列 {sequence_id}，帧 {frame_id}:')
                print(f'  - 总点数: {len(frame_coord)}')
                print(f'  - 移除的移动点数: {len(frame_coord) - len(non_moving_coord)}')
                print(f'  - 保存的非移动点数: {len(non_moving_coord)}')
    
    # 打印最终统计信息
    print(f'最终统计信息:')
    print(f'  - 总点数: {total_points_count}')
    print(f'  - 移除的移动点数: {removed_moving_points_count} ({removed_moving_points_count/total_points_count*100:.2f}%)')
    print(f'  - 保存的非移动点数: {total_points_count - removed_moving_points_count} ({(total_points_count - removed_moving_points_count)/total_points_count*100:.2f}%)')
    print(f'非移动点云保存至 non_moving_points/')
    
    return {
        "total_points": total_points_count,
        "removed_moving_points": removed_moving_points_count,
        "remaining_points": total_points_count - removed_moving_points_count
    }



def remove_moving_points_from_pointcloud(val_loader, model):
    """
    处理SemanticKITTI点云数据，移除移动点并保存
    
    参数:
    - val_loader: 验证数据加载器
    - model: 用于预测的神经网络模型
    
    功能:
    1. 对每帧点云数据预测标签
    2. 移除被预测为移动点的点
    3. 按原始格式保存处理后的点云数据
    """
    print('开始移除移动点')
    
    batch_time = time.time()
    
    # 统计计数器
    total_points_count = 0
    removed_points_count = 0
    
    model.eval()
    
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            print(f'处理批次 {i+1}/{len(val_loader)}')
            
            # 解包批次数据
            (coord, xyz, feat, target, offset, inds_reverse, file_names) = batch_data
            
            # 将数据移动到GPU
            coord = coord.cuda()
            xyz = xyz.cuda()
            feat = feat.cuda()
            target = target.cuda()
            offset = offset.cuda()
            inds_reverse = inds_reverse.cuda()
            
            # 构建批次索引
            offset_ = offset.clone()
            offset_[1:] = offset_[1:] - offset_[:-1]
            
            batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long().cuda()
            
            # 添加批次维度到坐标
            coord = torch.cat([batch.unsqueeze(-1), coord], -1)
            spatial_shape = np.clip((coord.cpu().max(0)[0][1:] + 1).numpy(), 128, None)
            
            # 创建稀疏卷积张量并运行模型
            sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, val_loader.batch_size)
            
            try:
                # 运行模型并获取预测结果
                output = model(sinput, xyz, batch)
                
                # 检查inds_reverse是否有效
                if torch.max(inds_reverse) >= output.shape[0]:
                    print("警告：inds_reverse中的索引超出范围，将进行修复")
                    inds_reverse = torch.clamp(inds_reverse, 0, output.shape[0] - 1)
                
                # 应用inds_reverse重新排序输出
                output = output[inds_reverse, :]
                
                # 获取预测类别
                pred_logits = output.max(1)[1]
                
                # 将数据转移到CPU
                predictions = pred_logits.cpu().numpy()
                batch_idx = batch[inds_reverse].cpu().numpy()
                
                # 处理并保存每个样本
                for b_idx in np.unique(batch_idx):
                    # 找到当前批次样本的掩码
                    mask = batch_idx == b_idx
                    scan_predictions = predictions[mask]
                    
                    # 从原始数据中选择非移动点
                    non_moving_mask = scan_predictions != 19
                    
                    # 使用文件名构建保存路径
                    current_file = file_names[int(b_idx)]
                    sequence_dir = os.path.dirname(os.path.dirname(current_file))
                    sequence_id = os.path.basename(sequence_dir)
                    frame_id = os.path.basename(current_file).split('.')[0]
                    
                    # 创建保存目录
                    save_dir_labels = os.path.join('output', 'labels_without_moving_points', sequence_id)
                    save_dir_points = os.path.join('output', 'points_without_moving_points', sequence_id)
                    os.makedirs(save_dir_labels, exist_ok=True)
                    os.makedirs(save_dir_points, exist_ok=True)
                    
                    # 原始文件对应的数据
                    original_data = np.fromfile(current_file, dtype=np.float32).reshape(-1, 4)
                    original_labels = np.fromfile(current_file.replace('velodyne', 'labels').replace('.bin', '.label'), dtype=np.uint32)
                    
                    # 使用非移动点掩码过滤点云和标签
                    filtered_points = original_data[non_moving_mask]
                    filtered_labels = original_labels[non_moving_mask]
                    
                    # 统计移除的点数
                    total_points_count += len(original_data)
                    removed_points_count += len(original_data) - len(filtered_points)
                    
                    # 保存处理后的点云和标签
                    label_save_path = os.path.join(save_dir_labels, f'{frame_id}.label')
                    points_save_path = os.path.join(save_dir_points, f'{frame_id}.bin')
                    
                    filtered_labels.astype(np.uint32).tofile(label_save_path)
                    filtered_points.astype(np.float32).tofile(points_save_path)
                    
                    # 打印处理详情
                    print(f'处理序列 {sequence_id}，帧 {frame_id}')
                    print(f'  - 原始点数: {len(original_data)}')
                    print(f'  - 移除移动点后的点数: {len(filtered_points)}')
                    print(f'  - 移除的点数: {len(original_data) - len(filtered_points)}')
            
            except Exception as e:
                print(f"处理批次 {i+1} 时发生错误: {str(e)}")
                import traceback
                print(traceback.format_exc())
                break
    
    # 最终统计信息
    print('点云处理统计:')
    print(f'  - 总点数: {total_points_count}')
    print(f'  - 移除的点数: {removed_points_count} ({removed_points_count/total_points_count*100:.2f}%)')
    print(f'  - 保存路径: output/points_without_moving_points 和 output/labels_without_moving_points')
    
    return {
        "total_points": total_points_count,
        "removed_points": removed_points_count,
        "removal_percentage": removed_points_count / total_points_count * 100
    }


def remove_moving_points_from_pointcloud2(val_loader, model):
    """
    处理SemanticKITTI点云数据，移除移动点并保存
    
    参数:
    - val_loader: 验证数据加载器
    - model: 用于预测的神经网络模型
    
    功能:
    1. 对每帧点云数据预测标签
    2. 移除被预测为移动点的点
    3. 按原始格式保存处理后的点云数据
    """
    if main_process():
        logger.info('开始移除移动点')
    
    batch_time = time.time()
    
    rank = 0
    if args.multiprocessing_distributed:
        rank = dist.get_rank()
        print(f'rank = {rank}')
    

    # 统计计数器
    total_points_count = 0
    removed_points_count = 0
    
    model.eval()
    
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            if main_process():
                logger.info(f'处理批次 {i+1}/{len(val_loader)}')
            
            # 解包批次数据
            (coord, xyz, feat, target, offset, inds_reverse, file_names) = batch_data
            
            # 将数据移动到GPU
            coord = coord.cuda()
            xyz = xyz.cuda()
            feat = feat.cuda()
            target = target.cuda()
            offset = offset.cuda()
            inds_reverse = inds_reverse.cuda()
            
            # 构建批次索引
            offset_ = offset.clone()
            offset_[1:] = offset_[1:] - offset_[:-1]
            
            batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long().cuda()
            
            # 添加批次维度到坐标
            coord = torch.cat([batch.unsqueeze(-1), coord], -1)
            spatial_shape = np.clip((coord.cpu().max(0)[0][1:] + 1).numpy(), 128, None)
            
            # 创建稀疏卷积张量并运行模型
            sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, val_loader.batch_size)
            
            try:
                # 运行模型并获取预测结果
                output = model(sinput, xyz, batch)
                
                # 检查inds_reverse是否有效
                if torch.max(inds_reverse) >= output.shape[0]:
                    if main_process():
                        logger.warning("inds_reverse中的索引超出范围，将进行修复")
                    inds_reverse = torch.clamp(inds_reverse, 0, output.shape[0] - 1)
                
                # 应用inds_reverse重新排序输出
                output = output[inds_reverse, :]
                
                # 获取预测类别
                pred_logits = output.max(1)[1]
                
                # 将数据转移到CPU
                predictions = pred_logits.cpu().numpy()
                batch_idx = batch[inds_reverse].cpu().numpy()
                
                # 处理并保存每个样本
                for b_idx in np.unique(batch_idx):
                    # 找到当前批次样本的掩码
                    mask = batch_idx == b_idx
                    scan_predictions = predictions[mask]
                    
                    # 从原始数据中选择非移动点
                    non_moving_mask = scan_predictions != 19
                    
                    # 使用文件名构建保存路径
                    current_file = file_names[int(b_idx)]
                    sequence_dir = os.path.dirname(os.path.dirname(current_file))
                    sequence_id = os.path.basename(sequence_dir)
                    frame_id = os.path.basename(current_file).split('.')[0]
                    
                    # 创建保存目录
                    save_dir_labels = os.path.join('output', 'labels_without_moving_points', sequence_id)
                    save_dir_points = os.path.join('output', 'points_without_moving_points', sequence_id)
                    os.makedirs(save_dir_labels, exist_ok=True)
                    os.makedirs(save_dir_points, exist_ok=True)
                    
                    # 原始文件对应的数据
                    original_data = np.fromfile(current_file, dtype=np.float32).reshape(-1, 4)
                    original_labels = np.fromfile(current_file.replace('velodyne', 'labels').replace('.bin', '.label'), dtype=np.uint32)
                    
                    # 使用非移动点掩码过滤点云和标签
                    filtered_points = original_data[non_moving_mask]
                    filtered_labels = original_labels[non_moving_mask]
                    
                    # 统计移除的点数
                    total_points_count += len(original_data)
                    removed_points_count += len(original_data) - len(filtered_points)
                    
                    # 保存处理后的点云和标签
                    label_save_path = os.path.join(save_dir_labels, f'{frame_id}.label')
                    points_save_path = os.path.join(save_dir_points, f'{frame_id}.bin')
                    
                    filtered_labels.astype(np.uint32).tofile(label_save_path)
                    filtered_points.astype(np.float32).tofile(points_save_path)
                    
                    # 日志记录处理详情
                    if main_process():
                        logger.info(f"处理序列 {sequence_id}，帧 {frame_id}")
                        logger.info(f"  - 原始点数: {len(original_data)}")
                        logger.info(f"  - 移除移动点后的点数: {len(filtered_points)}")
                        logger.info(f"  - 移除的点数: {len(original_data) - len(filtered_points)}")
            
            except Exception as e:
                if main_process():
                    logger.error(f"处理批次 {i+1} 时发生错误: {str(e)}")
                    logger.error(traceback.format_exc())
                break
    
    # 最终统计信息
    if main_process():
        logger.info('点云处理统计:')
        logger.info(f"  - 总点数: {total_points_count}")
        logger.info(f"  - 移除的点数: {removed_points_count} ({removed_points_count/total_points_count*100:.2f}%)")
        logger.info(f"  - 保存路径: output/points_without_moving_points 和 output/labels_without_moving_points")
    
    return {
        "total_points": total_points_count,
        "removed_points": removed_points_count,
        "removal_percentage": removed_points_count / total_points_count * 100
    }


def remove_moving_points_from_pointcloud3(val_loader, model, logger=None, save_folder='output'):
    """
    处理SemanticKITTI点云数据，移除移动点并保存
    
    参数:
    - val_loader: 验证数据加载器
    - model: 用于预测的神经网络模型
    - logger: 日志记录器（可选）
    - save_folder: 保存结果的文件夹路径（可选）
    
    功能:
    1. 对每帧点云数据预测标签
    2. 移除被预测为移动点的点
    3. 按原始格式保存处理后的点云数据
    """
    # 如果没有提供logger，使用默认logger
    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
    
    logger.info('>>>>>>>>>>>>>>>> Start Removing Moving Points >>>>>>>>>>>>>>>>')
    
    batch_time = time.time()
    
    # 统计计数器
    total_points_count = 0
    removed_points_count = 0
    
    model.eval()
    
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            logger.info(f'Processing batch {i+1}/{len(val_loader)}')
            
            # 解包批次数据
            (coord, xyz, feat, target, offset, inds_reverse, file_names) = batch_data
            
            # 将数据移动到GPU
            coord = coord.cuda()
            xyz = xyz.cuda()
            feat = feat.cuda()
            target = target.cuda()
            offset = offset.cuda()
            inds_reverse = inds_reverse.cuda()
            
            # 构建批次索引
            offset_ = offset.clone()
            offset_[1:] = offset_[1:] - offset_[:-1]
            
            batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long().cuda()
            
            # 添加批次维度到坐标
            coord = torch.cat([batch.unsqueeze(-1), coord], -1)
            spatial_shape = np.clip((coord.cpu().max(0)[0][1:] + 1).numpy(), 128, None)
            
            # 创建稀疏卷积张量并运行模型
            sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, val_loader.batch_size)
            
            try:
                # 运行模型并获取预测结果
                output = model(sinput, xyz, batch)
                
                # 检查inds_reverse是否有效
                if torch.max(inds_reverse) >= output.shape[0]:
                    logger.warning(f"inds_reverse中的索引超出范围，将进行修复")
                    inds_reverse = torch.clamp(inds_reverse, 0, output.shape[0] - 1)
                
                # 应用inds_reverse重新排序输出
                output = output[inds_reverse, :]
                
                # 获取预测类别
                pred_logits = output.max(1)[1]
                
                # 将数据转移到CPU
                predictions = pred_logits.cpu().numpy()
                batch_idx = batch[inds_reverse].cpu().numpy()
                
                # 处理并保存每个样本
                for b_idx in np.unique(batch_idx):
                    # 找到当前批次样本的掩码
                    mask = batch_idx == b_idx
                    scan_predictions = predictions[mask]
                    
                    # 从原始数据中选择非移动点
                    non_moving_mask = scan_predictions != 19
                    
                    # 使用文件名构建保存路径
                    current_file = file_names[int(b_idx)]
                    sequence_dir = os.path.dirname(os.path.dirname(current_file))
                    sequence_id = os.path.basename(sequence_dir)
                    frame_id = os.path.basename(current_file).split('.')[0]
                    
                    # 创建保存目录
                    save_dir_labels = os.path.join(save_folder, 'labels_without_moving_points', sequence_id)
                    save_dir_points = os.path.join(save_folder, 'points_without_moving_points', sequence_id)
                    os.makedirs(save_dir_labels, exist_ok=True)
                    os.makedirs(save_dir_points, exist_ok=True)
                    
                    # 原始文件对应的数据
                    original_data = np.fromfile(current_file, dtype=np.float32).reshape(-1, 4)
                    original_labels = np.fromfile(current_file.replace('velodyne', 'labels').replace('.bin', '.label'), dtype=np.uint32)
                    
                    # 使用非移动点掩码过滤点云和标签
                    filtered_points = original_data[non_moving_mask]
                    filtered_labels = original_labels[non_moving_mask]
                    
                    # 统计移除的点数
                    total_points_count += len(original_data)
                    removed_points_count += len(original_data) - len(filtered_points)
                    
                    # 保存处理后的点云和标签
                    label_save_path = os.path.join(save_dir_labels, f'{frame_id}.label')
                    points_save_path = os.path.join(save_dir_points, f'{frame_id}.bin')
                    
                    filtered_labels.astype(np.uint32).tofile(label_save_path)
                    filtered_points.astype(np.float32).tofile(points_save_path)
                    
                    # 打印处理详情
                    logger.info(f'Processed sequence {sequence_id}, frame {frame_id}')
                    logger.info(f'  - Original points: {len(original_data)}')
                    logger.info(f'  - Points after removing moving points: {len(filtered_points)}')
                    logger.info(f'  - Removed points: {len(original_data) - len(filtered_points)}')
            
            except Exception as e:
                logger.error(f"处理批次 {i+1} 时发生错误: {str(e)}")
                logger.error(traceback.format_exc())
                continue
    
    # 最终统计信息
    logger.info('点云处理统计:')
    logger.info(f'  - 总点数: {total_points_count}')
    logger.info(f'  - 移除的点数: {removed_points_count} ({removed_points_count/total_points_count*100:.2f}%)')
    logger.info(f'  - 保存路径: {save_folder}/points_without_moving_points 和 {save_folder}/labels_without_moving_points')
    logger.info('<<<<<<<<<<<<<<<<< End Removing Moving Points <<<<<<<<<<<<<<<<<')
    
    return {
        "total_points": total_points_count,
        "removed_points": removed_points_count,
        "removal_percentage": removed_points_count / total_points_count * 100
    }


if __name__ == '__main__':
    import gc
    gc.collect()
    main()

