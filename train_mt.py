import argparse
import logging
import yaml
import math
import os
import random
import time
from pathlib import Path
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from utils.general import init_seeds, set_logging, check_git_status, check_requirements, check_file, \
    increment_path, get_latest_run
from utils.torch_utils import select_device
from utils.trainer import MTeacherTrainer

logger = logging.getLogger(__name__)


class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch Detecion.')
        # model and dataset
        parser.add_argument('--data', type=str, default='data/ipsc_yolo_semi.yaml', help='data.yaml path')
        parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
        parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
        parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
        parser.add_argument('--quad', action='store_true', help='quad dataloader')
        parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')

        # training hyper params
        parser.add_argument('--unsup-loss-weight', type=float, default=0.5, help='Weight for unsupervised weight')
        parser.add_argument('--linear-lr', action='store_true', help='linear LR')
        parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
        parser.add_argument('--rect', action='store_true', help='rectangular training')
        parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
        parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs')
        parser.add_argument('--epochs', type=int, default=200, help='Number of total epochs, including burn-in epochs.')
        parser.add_argument('--ema-rate', type=float, default=0.9996, help='Weight for teacher model in mean-teacher')
        parser.add_argument('--burnin-epochs', type=int, default=80, help='Burn in epochs before mean-teacher training')
        parser.add_argument('--iou-thresh-mt', type=float, default=0.6, help='iou thresh for generating pseudo label')
        parser.add_argument('--conf_thresh-mt', type=float, default=0.8, help='confidence thresh for pseudo label.')
        parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
        parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
        parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
        parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
        parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
        parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
        parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')

        # cuda, seed and save
        parser.add_argument('--log-artifacts', action='store_true', help='log artifacts, i.e. final trained model')
        parser.add_argument('--log-imgs', type=int, default=16, help='number of images for W&B logging, max 100')
        parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
        parser.add_argument('--device', default='5', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--notest', action='store_true', help='only test final epoch')
        parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--project', default='runs/semi_train', help='save to project/name')
        parser.add_argument('--name', default='exp', help='save to project/name')
        parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
        parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')

        # evaluation option
        parser.add_argument('--val-interval', type=int, default=1,
                            help='Number of epoches between valing')
        parser.add_argument('--run-test', action='store_true', default=False,
                            help='Run test after training.')
        # the parser
        self.parser = parser

    def parse(self):
        opt = self.parser.parse_args()
        # Set DDP variables
        opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
        opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
        set_logging(opt.global_rank)
        if opt.global_rank in [-1, 0]:
            check_git_status()
            check_requirements()
        # Resume
        if opt.resume:  # resume an interrupted run
            ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
            assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
            with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
                opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))  # replace
            opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = \
                '', ckpt, True, opt.total_batch_size, opt.global_rank, opt.local_rank  # reinstate
            logger.info('Resuming training from %s' % ckpt)
        else:
            # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
            opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(
                opt.hyp)  # check files
            assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
            opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
            opt.name = 'evolve' if opt.evolve else opt.name
            # increment run
            opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)
        # DDP mode
        opt.total_batch_size = opt.batch_size
        device = select_device(opt.device, batch_size=opt.batch_size)
        if opt.local_rank != -1:
            assert torch.cuda.device_count() > opt.local_rank
            torch.cuda.set_device(opt.local_rank)
            device = torch.device('cuda', opt.local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
            assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
            opt.batch_size = opt.total_batch_size // opt.world_size

        # Hyperparameters
        with open(opt.hyp) as f:
            hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

        # Train
        logger.info(opt)
        tb_writer = None
        if not opt.evolve:
            # tb_writer = None  # init loggers
            if opt.global_rank in [-1, 0]:
                logger.info(
                    f'Start Tensorboard with "tensorboard --logdir {opt.save_dir}", view at http://localhost:6006/')
                tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
            # TODO: change by wang001
            # opt.device = device
            # opt.logger = logger
        else:
            print('Still not support hyper-parameters evolve!')
        # print(args)
        return opt, hyp, device, tb_writer, logger


if __name__ == '__main__':
    # init_seeds(seed=1109)
    opt,hyp, device, tb_writer, logger = Options().parse()
    init_seeds(2 + opt.global_rank)
    trainer = MTeacherTrainer(opt, hyp, device, tb_writer, logger)
    trainer.train()
    # training

    # TODO: recording the number of pseudo label generated during training time
    # TODO: adding dynamic confidence score for pseudo label
    # TODO: don't compute box_loss for pseudo label

