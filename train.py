import argparse
from model.data.transforms.data_preprocess import DummyTargetTransform, TrainTransform
import os
import shutil
import numpy as np
import datetime

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR

from model.config import cfg
from model.engine.trainer import do_pretrain_for_mp, do_train
from model.modeling.build_model import ModelWithLoss
from model.data.samplers import IterationBasedBatchSampler
from model.utils.sync_batchnorm import convert_model
from model.utils.misc import str2bool, fix_model_state_dict, worker_init_fn
from model.data.dataset import build_dataset
from model.data.transforms.transforms import MakeHeatmap
from model.data.transforms.data_preprocess import *


def parse_args():
    parser = argparse.ArgumentParser(description='pytorch training code')
    parser.add_argument('--config_file', type=str, default='', metavar='FILE', help='path to config file')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--output_dirname', type=str, default='', help='')
    parser.add_argument('--log_step', type=int, default=50, help='')
    parser.add_argument('--eval_step', type=int, default=0, help='')
    parser.add_argument('--save_step', type=int, default=50000, help='')
    parser.add_argument('--num_gpus', type=int, default=1, help='')
    parser.add_argument('--num_workers', type=int, default=16, help='')
    parser.add_argument('--resume_iter', type=int, default=0, help='')
    parser.add_argument('--gradient_logging', action='store_true')

    return parser.parse_args()


def train(args, cfg):
    model = ModelWithLoss(cfg).to('cuda')
    if cfg.SOLVER.SYNC_BATCHNORM:
        model = convert_model(model).to('cuda')
    print('------------Model Architecture-------------')
    print(model)

    print('Loading Datasets...')
    data_loader = {}
    # BOX_THRESHOLD [0.3, 0.95, 0.95], DOWN_RATIOS = [4, 2, 1]
    train_transforms = [TrainTransform(cfg, cfg.MODEL.DOWN_RATIOS[i], cfg.SOLVER.DATA.BOX_THRESHOLD[i]) if i in cfg.MODEL.VALID_SCALE else DummyTransform() for i in range(len(cfg.MODEL.DOWN_RATIOS))]
    # train_transforms = [TrainTransform(cfg, down_ratio) for down_ratio in cfg.MODEL.DOWN_RATIOS]
    train_target_transforms = [MakeHeatmap(cfg, cfg.MODEL.DOWN_RATIOS[i]) if i in cfg.MODEL.VALID_SCALE else DummyTargetTransform() for i in range(len(cfg.MODEL.DOWN_RATIOS))]
    # train_target_transforms = [MakeHeatmap(cfg.MODEL.NUM_CLASSES, cfg.SOLVER.DATA.MAX_OBJECTS, down_ratio) for down_ratio in cfg.MODEL.DOWN_RATIOS]
    train_dataset = build_dataset(cfg, dataset_list=cfg.DATASET.TRAIN, transform=train_transforms, target_transform=train_target_transforms)
    sampler = RandomSampler(train_dataset)
    batch_sampler = BatchSampler(sampler=sampler, batch_size=cfg.SOLVER.BATCH_SIZE, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iterations=cfg.SOLVER.MAX_ITER)
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_sampler=batch_sampler, pin_memory=True, worker_init_fn=worker_init_fn)

    data_loader['train'] = train_loader

    if args.eval_step != 0:
        val_transforms = None
        val_dataset = None
        sampler = SequentialSampler(val_dataset)
        batch_sampler = BatchSampler(sampler=sampler, batch_size=args._eval_batchsize, drop_last=False)
        val_loader = DataLoader(val_dataset, num_workers=args.num_workers, batch_sampler=batch_sampler)

        data_loader['val'] = val_loader

    optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, list(model.extractors.parameters()) + list(model.detector.parameters())), lr=cfg.SOLVER.DETECTOR.LR)
    scheduler = MultiStepLR(optimizer, cfg.SOLVER.LR_DECAY)
    d_optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, model.discriminator.parameters()), lr=cfg.SOLVER.DISCRIMINATOR.LR)
    d_scheduler = MultiStepLR(d_optimizer, cfg.SOLVER.LR_DECAY)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.MIXED_PRECISION)

    if args.resume_iter != 0:
        print('Resume from {}'.format(os.path.join(cfg.OUTPUT_DIR, 'model', 'iteration_{}.pth'.format(args.resume_iter))))
        model.load_state_dict(fix_model_state_dict(torch.load(os.path.join(cfg.OUTPUT_DIR, 'model', 'iteration_{}.pth'.format(args.resume_iter)))))
        optimizer.load_state_dict(torch.load(os.path.join(cfg.OUTPUT_DIR, 'optimizer', 'iteration_{}.pth'.format(args.resume_iter))))
        scaler.load_state_dict(torch.load(os.path.join(cfg.OUTPUT_DIR, 'scaler', 'iteration_{}.pth'.format(args.resume_iter))))
   
    model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpus)))

    if not args.debug:
        summary_writer = SummaryWriter(log_dir=cfg.OUTPUT_DIR)
    else:
        summary_writer = None

    if cfg.MIXED_PRECISION:
        do_pretrain_for_mp(args, cfg, model, optimizer, scheduler, d_optimizer, d_scheduler, data_loader, summary_writer)

    do_train(args, cfg, model, optimizer, scheduler, d_optimizer, d_scheduler, scaler, data_loader, summary_writer)


def main():
    args = parse_args()
    
    if len(args.config_file) > 0:
        print('Configration file is loaded from {}'.format(args.config_file))
        cfg.merge_from_file(args.config_file)
    
    if len(args.output_dirname) == 0:
        dt_now = datetime.datetime.now()
        output_dirname = os.path.join('output', str(dt_now.date()) + '_' + str(dt_now.time()))
    else:
        output_dirname = args.output_dirname
    cfg.OUTPUT_DIR = output_dirname
    cfg.freeze()

    assert torch.cuda.is_available(), 'GPU not found'

    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)

    torch.backends.cudnn.benchmark = True
    
    # torch.backends.cudnn.deterministic = True
    # torch.autograd.set_detect_anomaly(True)
    
    if not args.debug:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        if not len(args.config_file) == 0:
            shutil.copy2(args.config_file, os.path.join(cfg.OUTPUT_DIR, 'config.yaml'))

    train(args, cfg)


if __name__ == '__main__':
    main()
