import os
import time
import yaml
import math
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from threading import Thread
from collections import OrderedDict

import test  # import test.py to get mAP after each epoch
from models.yolo import Model
from utils.loss import ComputeLoss
from utils.autoanchor import check_anchors
from models.experimental import attempt_load
from utils.google_utils import attempt_download
from utils.plots import plot_labels, plot_images
from utils.datasets import create_dataloader_trian_mt, create_dataloader
from utils.general import colorstr, one_cycle, check_img_size, check_dataset, labels_to_class_weights, \
    labels_to_image_weights, fitness, non_max_suppression, xyxy2xywh
from utils.torch_utils import torch_distributed_zero_first, intersect_dicts, ModelEMA, is_parallel

import torch
import torch.nn as nn
from torch.cuda import amp
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP


# fully supervised training
class BaselineTrainer:
    def __init__(self, args):
        pass


# mean-teacher supervised trainer
class MTeacherTrainer:
    def __init__(self, opt, hyp, device, tb_writer, logger):
        # logging & tensorboard
        self.opt = opt
        self.hyp = hyp
        self.logger = logger
        self.tb_writer = tb_writer

        # hyper-parameters
        self.lr = None
        self.lf = None
        self.nw = None
        self.ema = None
        self.scaler = None
        self.optimizer = None
        self.scheduler = None
        self.epoch = 0
        self.hyp_dic = hyp
        self.start_epoch = 0
        self.epochs = opt.epochs
        self.evolve = opt.evolve
        self.ema_rate = opt.ema_rate  # ratio of keeping weight of teacher model
        self.batch_size = opt.batch_size
        self.burnin_epochs = opt.burnin_epochs
        self.iou_thresh_mt = opt.iou_thresh_mt
        self.conf_thresh_mt = opt.conf_thresh_mt
        self.with_pseudo_label = None
        self.total_batch_size = opt.total_batch_size
        self.unsup_loss_weight = opt.unsup_loss_weight

        # save & other
        self.device = device
        self.best_fitness = 0.0
        self.best_fitness_mt = 0.0
        self.rank = opt.global_rank
        self.plots = not self.evolve  # create plots
        self.save_dir = Path(opt.save_dir)
        self.cuda = self.device.type != 'cpu'

        # for initialize, leave alone
        self.was_initialized = False

        # dataset
        self.gs = None
        self.nc = None
        self.nl = None
        self.nb = None
        self.nb_u = None
        self.nbs = None
        self.mlc = None  # max label class
        self.imgsz = None
        self.names = None
        self.testloader = None
        self.imgsz_test = None
        self.train_path_l = None
        self.train_path_u = None
        self.val_path = None
        self.test_path = None
        self.dataset_l = None
        self.dataloader_l = None
        self.dataloader_u = None
        self.data = opt.data
        with open(self.data) as f:
            self.data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
        with torch_distributed_zero_first(self.rank):
            check_dataset(self.data_dict)  # check

        self.nc = 1 if opt.single_cls else int(self.data_dict['nc'])  # number of classes
        # TODO: add dataloader for mean-teacher SSL

        # model
        self.model = None
        self.accumulate = None
        self.teacher_model = None
        self.weights = opt.weights
        self.pretrained = self.weights.endswith('.pt')

        # loss & metrics
        self.maps = None
        self.results_un = None
        self.results_sup = None
        self.compute_loss_sup = None
        self.mloss_buruin = None
        self.mloss_mt = None

        # Save run settings
        self.w_dir = os.path.join(self.save_dir, 'weights') + os.sep
        os.makedirs(self.w_dir, exist_ok=True)
        self.weight_best_sup = os.path.join(self.w_dir, 'best_sup.pt')
        self.weight_best_un = os.path.join(self.w_dir, 'best_un.pt')
        self.weight_last = os.path.join(self.w_dir, 'last.pt')
        self.results_file = os.path.join(self.save_dir, 'results.txt')
        with open(self.save_dir / 'hyp.yaml', 'w') as f:
            yaml.dump(self.hyp_dic, f, sort_keys=False)
        with open(self.save_dir / 'opt.yaml', 'w') as f:
            yaml.dump(vars(opt), f, sort_keys=False)

    def initialize(self):
        self.initialize_network_optim()
        # initialize_lr_scheduler
        self.initilize_lr_scheduler()
        self.initialize_dataloader()
        self.was_initialized = True

    def train(self):
        self.logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in self.hyp_dic.items()))
        # initialize model, optimizer , lr_scheduler, data_loader
        if not self.was_initialized:
            self.initialize()

        # Start training
        t0 = time.time()
        self.train_loop(self.start_epoch, self.epochs)
        # TODO: change the number of end epochs
        self.logger.info(
            '%g epochs completed in %.3f hours.\n' % (self.epoch - self.start_epoch + 1, (time.time() - t0) / 3600))

        # init_seeds(2 + self.rank)

    def train_loop(self, start_epoch, max_epoch):
        # number of warmup iterations, max(3 epochs, 1k iterations)
        self.nw = max(round(self.hyp['warmup_epochs'] * self.nb), 1000)
        self.maps = np.zeros(self.nc)
        self.results_sup = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        self.scheduler.last_epoch = start_epoch - 1  # do not move
        self.scaler = amp.GradScaler(enabled=self.cuda)
        self.compute_loss_sup = ComputeLoss(self.model)
        self.logger.info(f'Image sizes {self.imgsz} train, {self.imgsz_test} test\n'
                         f'Using {self.dataloader_l.num_workers} dataloader workers\n'
                         f'Logging results to {self.save_dir}\n'
                         f'Starting training for {self.epochs} epochs...')
        # TODO: burn-in training
        for self.epoch in range(start_epoch, max_epoch):
            self.model.train()
            self.run_step_full_semisup()
        pass

    def run_step_full_semisup(self):
        # burn-in stage
        if self.epoch < self.burnin_epochs:
            # perform burn-in operation with student model and labeled dataset
            # Update image weights (optional)
            if self.opt.image_weights:
                # Generate indices
                if self.rank in [-1, 0]:
                    cw = self.model.class_weights.cpu().numpy() * (1 - self.maps) ** 2 / self.nc  # class weights
                    iw = labels_to_image_weights(self.dataset_l.labels, nc=self.nc, class_weights=cw)  # image weights
                    self.dataset_l.indices = random.choices(range(self.dataset_l.n), weights=iw,
                                                            k=self.dataset_l.n)  # rand weighted idx
                # Broadcast if DDP
                if self.rank != -1:
                    indices = (torch.tensor(self.dataset_l.indices) if self.rank == 0
                               else torch.zeros(self.dataset_l.n)).int()
                    dist.broadcast(indices, 0)
                    if self.rank != 0:
                        self.dataset_l.indices = indices.cpu().numpy()

            self.mloss_buruin = torch.zeros(4, device=self.device)  # mean loss
            if self.rank != -1:
                self.dataloader_l.sampler.set_epoch(self.epoch)
            pbar_burnin = enumerate(self.dataloader_l)
            self.logger.info(('\n' + '%10s' * 8) % ('Warm epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total',
                                                    'targets', 'img_size'))
            if self.rank in [-1, 0]:
                pbar_burnin = tqdm(pbar_burnin, total=self.nb)  # progress bar
            self.optimizer.zero_grad()
            for i, (imgs, targets, paths, _) in pbar_burnin:  # batch ------------------------------------------------
                ni = i + self.nb * self.epoch  # number integrated batches (since train start)
                imgs = imgs.to(self.device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

                # Warmup
                if ni <= self.nw:
                    xi = [0, self.nw]  # x interp
                    # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                    accumulate = max(1, np.interp(ni, xi, [1, self.nbs / self.total_batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, xi,
                                            [self.hyp['warmup_bias_lr'] if j == 2 else 0.0,
                                             x['initial_lr'] * self.lf(self.epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.hyp['warmup_momentum'], self.hyp['momentum']])

                # Multi-scale
                if self.opt.multi_scale:
                    sz = random.randrange(self.imgsz * 0.5, self.imgsz * 1.5 + self.gs) // self.gs * self.gs  # size
                    sf = sz / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / self.gs) * self.gs for x in
                              imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                        imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

                # Forward
                with amp.autocast(enabled=self.cuda):
                    pred = self.model(imgs)  # forward
                    loss, loss_items = self.compute_loss_sup(pred, targets.to(self.device))  # loss scaled by batch_size
                    if self.rank != -1:
                        loss *= self.opt.world_size  # gradient averaged between devices in DDP mode
                    if self.opt.quad:
                        loss *= 4.

                # Backward
                self.scaler.scale(loss).backward()

                # Optimize
                if ni % self.accumulate == 0:
                    self.scaler.step(self.optimizer)  # optimizer.step
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    if self.ema:
                        self.ema.update(self.model)

                # Print
                if self.rank in [-1, 0]:
                    self.mloss_buruin = (self.mloss_buruin * i + loss_items) / (i + 1)  # update mean losses
                    mem = '%.3gG' % (
                        torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                    s = ('%10s' * 2 + '%10.4g' * 6) % (
                        '%g/%g' % (self.epoch, self.epochs - 1), mem, *self.mloss_buruin, targets.shape[0],
                        imgs.shape[-1])
                    pbar_burnin.set_description(s)

                    # Plot
                    if self.plots and ni < 3:
                        f = self.save_dir / f'train_batch{ni}.jpg'  # filename
                        Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()

                # end batch --------------------------------------------------------------------------
            # end epoch ----------------------------------------------------------------------------------

            # Scheduler
            self.lr = [x['lr'] for x in self.optimizer.param_groups]  # for tensorboard
            self.scheduler.step()

            # DDP process 0 or single-GPU
            if self.rank in [-1, 0]:
                # mAP
                if self.ema:
                    self.ema.update_attr(self.model,
                                         include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
                final_epoch = self.epoch + 1 == self.epochs
                if not self.opt.notest or final_epoch:  # Calculate mAP
                    self.results_sup, maps, times = test.test(self.opt.data,
                                                              batch_size=self.batch_size * 2,
                                                              imgsz=self.imgsz_test,
                                                              model=self.ema.ema,
                                                              single_cls=self.opt.single_cls,
                                                              dataloader=self.testloader,
                                                              save_dir=self.save_dir,
                                                              verbose=self.nc < 50 and final_epoch,
                                                              plots=self.plots and final_epoch,
                                                              compute_loss=self.compute_loss_sup)
                # Write
                with open(self.results_file, 'a') as f:
                    # s + P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
                    f.write(s + '%10.4g' * 7 % self.results_sup + '\n')
                if len(self.opt.name) and self.opt.bucket:
                    os.system('gsutil cp %s gs://%s/results/results%s.txt' %
                              (self.results_file, self.opt.bucket, self.opt.name))

                # Log
                tags = ['train_burnin/box_loss', 'train_burnin/obj_loss', 'train_burnin/cls_loss',  # train loss
                        'metrics_burnin/precision', 'metrics_burnin/recall', 'metrics_burnin/mAP_0.5',
                        'metrics_burnin/mAP_0.5:0.95', 'val_burnin/box_loss', 'val_burnin/obj_loss',  # val loss
                        'val_burnin/cls_loss', 'x_burnin/lr0', 'x_burnin/lr1', 'x_burnin/lr2']  # params
                for x, tag in zip(list(self.mloss_buruin[:-1]) + list(self.results_sup) + self.lr, tags):
                    if self.tb_writer:
                        self.tb_writer.add_scalar(tag, x, self.epoch)  # tensorboard

                # Update best mAP
                # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
                # TODO: save best model for burn-in stage
                fi = fitness(np.array(self.results_sup).reshape(1, -1))
                if fi > self.best_fitness:
                    self.best_fitness = fi

                # Save  best model for burn-in stage
                save = not self.opt.nosave
                if save:
                    with open(self.results_file, 'r') as f:  # create checkpoint
                        ckpt = {'epoch': self.epoch,
                                'best_fitness': self.best_fitness,
                                'training_results': f.read(),
                                'model': self.ema.ema,
                                'optimizer': None if final_epoch else self.optimizer.state_dict()}

                    if self.best_fitness == fi:
                        torch.save(ckpt, self.weight_best_sup)
                    del ckpt
            # end batch --------------------------------------------------------------------------

        else:
            # re-initialize lr scheduler for teacher-student mutual training
            self.initilize_lr_scheduler()
            self.mloss_mt = torch.zeros(4, device=self.device)  # mean loss

            if self.epoch == self.burnin_epochs:
                with torch_distributed_zero_first(self.rank):
                # if self.rank in [-1, 0]:
                    ckpt = torch.load(self.weight_best_sup, map_location=self.device)  # load checkpoint
                    # self.teacher_model = Model(self.opt.cfg or ckpt['model'].yaml, ch=3, nc=self.nc).to(self.device)
                    state_dict = ckpt['model'].float().state_dict()  # to FP32
                    state_dict = intersect_dicts(state_dict, self.teacher_model.state_dict())  # intersect
                    self.teacher_model.load_state_dict(state_dict, strict=False)  # load
                    self.logger.info('Transferred %g/%g items from %s' %
                                     (len(state_dict), len(self.teacher_model.state_dict()), self.weight_best_sup))
                    del ckpt, state_dict
                    # self._update_teacher_model(keep_rate=0.00)

            # else:
            #     self._update_teacher_model(keep_rate=self.ema_rate)

            if self.rank != -1:
                self.dataloader_l.sampler.set_epoch(self.epoch)

            self.logger.info(('\n' + '%10s' * 8) % ('MT epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total',
                                                    'targets', 'img_size'))

            self.nb_u = len(self.dataloader_u)
            if self.nb_u > self.nb:
                progress_bar = enumerate(self.dataloader_u)
                if self.rank in [-1, 0]:
                    progress_bar = tqdm(progress_bar, total=self.nb_u)
            else:
                progress_bar = enumerate(self.dataloader_l)
                if self.rank in [-1, 0]:
                    progress_bar = tqdm(progress_bar, total=self.nb)

            for i, pack in progress_bar:
                ni = i + self.nb * self.epoch  # number integrated batches (since train start)
                # for iPSC detection only, un_lableled data may be less than labeled data
                if self.nb_u > self.nb:
                    imgs_weak, imgs_str, paths = pack

                    try:
                        imgs, targets, paths, _ = next(dataloader_l_iter)
                    except:
                        dataloader_l_iter = iter(self.dataloader_l)
                        imgs, targets, paths, _ = next(dataloader_l_iter)
                else:
                    imgs, targets, paths, _ = pack
                    try:
                        imgs_weak, imgs_str, paths = next(dataloader_u_iter)
                    except:
                        dataloader_u_iter = iter(self.dataloader_u)
                        imgs_weak, imgs_str, paths = next(dataloader_u_iter)

                #  generate the pseudo-label using teacher model
                with torch.no_grad():
                    imgs_weak = imgs_weak.to(self.device, non_blocking=True)

                    # Half
                    half = self.device.type != 'cpu'  # half precision only supported on CUDA
                    if half:
                        self.teacher_model.half()
                        imgs_weak = imgs_weak.half()

                    self.teacher_model.eval()
                    inf_out, _ = self.teacher_model(imgs_weak)
                    output = non_max_suppression(inf_out, conf_thres=self.conf_thresh_mt, iou_thres=self.iou_thresh_mt,
                                                 labels=[])

                    # format pseudo label to target
                    label_index = [x for x, y in list(enumerate(output)) if len(y) != 0]
                    if len(label_index) == 0:
                        self.with_pseudo_label = False
                    else:
                        self.with_pseudo_label = True
                        pseudo_targets = []
                        gn = torch.tensor((imgs_weak.shape[2], imgs_weak.shape[3]))[[1, 0, 1, 0]]
                        img_idx = 0
                        for si, pred in enumerate(output):
                            if len(pred) == 0:
                                continue
                            else:
                                target_p = torch.zeros(len(pred), 6, dtype=torch.float32)
                                target_p[:, 0] = img_idx
                                for ii, (*xyxy, _, cls) in enumerate(pred):
                                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1)
                                    target_p[ii, 2:6] = xywh
                                    target_p[ii, 1] = cls

                                img_idx += 1
                                pseudo_targets.append(target_p)
                        # cat label together
                        pseudo_targets = torch.cat(pseudo_targets, 0)

                        # select strong augmented iamge with pseudo label and move to device
                        imgs_str = imgs_str[label_index]
                        imgs_str = imgs_str.to(self.device, non_blocking=True)

                # Forward
                with amp.autocast(enabled=self.cuda):
                    # pseudo label data forward
                    if self.with_pseudo_label:
                        pred_pseudo = self.model(imgs_str)  # forward
                        # loss scaled by batch_size
                        loss_pseudo, loss_pseudo_items = self.compute_loss_sup(pred_pseudo,
                                                                               pseudo_targets.to(self.device))
                        if self.rank != -1:
                            loss_pseudo *= self.opt.world_size  # gradient averaged between devices in DDP mode

                    else:
                        loss_pseudo = torch.zeros(1).to(self.device)
                        loss_pseudo_items = torch.zeros(4).to(self.device)

                    # labeled data forward
                    imgs = imgs.to(self.device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
                    pred = self.model(imgs)  # forward
                    loss_l, loss_l_items = self.compute_loss_sup(pred,
                                                                 targets.to(self.device))  # loss scaled by batch_size
                    if self.rank != -1:
                        loss_l *= self.opt.world_size  # gradient averaged between devices in DDP mode
                    if self.opt.quad:
                        loss_l *= 4.

                    total_loss = loss_l + self.unsup_loss_weight * loss_pseudo
                    total_loss_items =self.unsup_loss_weight * loss_pseudo_items + loss_l_items

                # Backward
                self.scaler.scale(total_loss).backward()

                # Optimize
                if ni % self.accumulate == 0:
                    self.scaler.step(self.optimizer)  # optimizer.step
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    if self.ema:
                        self.ema.update(self.model)

                # Print
                if self.rank in [-1, 0]:
                    self.mloss_mt = (self.mloss_mt * i + total_loss_items) / (i + 1)  # update mean losses
                    mem = '%.3gG' % (
                        torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                    s = ('%10s' * 2 + '%10.4g' * 6) % (
                        '%g/%g' % (self.epoch, self.epochs - 1), mem, *self.mloss_mt, targets.shape[0],
                        imgs.shape[-1])
                    progress_bar.set_description(s)

                # end batch --------------------------------------------------------------------------
            # end epoch ------------------------------------------------------------------------------

            # Scheduler
            self.lr = [x['lr'] for x in self.optimizer.param_groups]  # for tensorboard
            self.scheduler.step()

            # DDP process 0 or single-GPU
            if self.rank in [-1, 0]:
                # mAP
                if self.ema:
                    self.ema.update_attr(self.model,
                                         include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])

            # update teacher model after update student model for testing
            self._update_teacher_model(keep_rate=self.ema_rate)

            # DDP process 0 or single-GPU
            if self.rank in [-1, 0]:
                final_epoch = self.epoch + 1 == self.epochs
                if not self.opt.notest or final_epoch:  # Calculate mAP
                    self.results_un, maps, times = test.test(self.opt.data,
                                                             batch_size=self.batch_size * 2,
                                                             imgsz=self.imgsz_test,
                                                             model=self.teacher_model,
                                                             single_cls=self.opt.single_cls,
                                                             dataloader=self.testloader,
                                                             save_dir=self.save_dir,
                                                             verbose=self.nc < 50 and final_epoch,
                                                             plots=self.plots and final_epoch,
                                                             compute_loss=self.compute_loss_sup)
                # Write
                with open(self.results_file, 'a') as f:
                    # s + P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
                    if self.epoch == self.burnin_epochs:
                        f.write('mean teacher training...\n')
                    f.write(s + '%10.4g' * 7 % self.results_un + '\n')
                if len(self.opt.name) and self.opt.bucket:
                    os.system('gsutil cp %s gs://%s/results/results%s.txt' %
                              (self.results_file, self.opt.bucket, self.opt.name))

                # Log
                tags = ['train_mt/box_loss', 'train_mt/obj_loss', 'train_mt/cls_loss',  # train loss
                        'metrics_mt/precision', 'metrics_mt/recall', 'metrics_mt/mAP_0.5',
                        'metrics_mt/mAP_0.5:0.95', 'val_mt/box_loss', 'val_mt/obj_loss',  # val loss
                        'val_mt/cls_loss', 'x_mt/lr0', 'x_mt/lr1', 'x_mt/lr2']  # params
                for x, tag in zip(list(self.mloss_mt[:-1]) + list(self.results_un) + self.lr, tags):
                    if self.tb_writer:
                        self.tb_writer.add_scalar(tag, x, self.epoch)  # tensorboard

                fi = fitness(np.array(self.results_un).reshape(1, -1))
                if fi > self.best_fitness_mt:
                    self.best_fitness_mt = fi

                # Save  best model for burn-in stage
                save = not self.opt.nosave
                if save:
                    with open(self.results_file, 'r') as f:  # create checkpoint
                        ckpt = {'epoch': self.epoch,
                                'best_fitness': self.best_fitness_mt,
                                'training_results': f.read(),
                                'model': self.teacher_model,
                                'optimizer': None if final_epoch else self.optimizer.state_dict()}

                    if self.best_fitness_mt == fi:
                        torch.save(ckpt, self.weight_best_un)
                    del ckpt

    def initialize_network_optim(self):
        if self.pretrained:
            with torch_distributed_zero_first(self.rank):
                attempt_download(self.weights)  # download if not found locally
            ckpt = torch.load(self.weights, map_location=self.device)  # load checkpoint
            if self.hyp_dic.get('anchors'):
                ckpt['model'].yaml['anchors'] = round(self.hyp_dic['anchors'])  # force autoanchor
            self.model = Model(self.opt.cfg or ckpt['model'].yaml, ch=3, nc=self.nc).to(self.device)  # create
            self.teacher_model = Model(self.opt.cfg or ckpt['model'].yaml, ch=3, nc=self.nc).to(self.device)  # create
            exclude = ['anchor'] if self.opt.cfg or self.hyp_dic.get('anchors') else []  # exclude keys
            state_dict = ckpt['model'].float().state_dict()  # to FP32
            state_dict = intersect_dicts(state_dict, self.model.state_dict(), exclude=exclude)  # intersect
            self.model.load_state_dict(state_dict, strict=False)  # load
            self.logger.info(
                'Transferred %g/%g items from %s' % (
                    len(state_dict), len(self.model.state_dict()), self.weights))  # report
            del exclude
        else:
            self.model = Model(self.opt.cfg, ch=3, nc=self.nc).to(self.device)  # create
            self.teacher_model = Model(self.opt.cfg, ch=3, nc=self.nc).to(self.device)  # create teacher model

        # Freeze
        freeze = []  # parameter names to freeze (full or partial)
        # freeze = ['model.%s.' % x for x in range(10, 24)]  # parameter names to freeze (full or partial)
        for k, v in self.model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                print('freezing %s' % k)
                v.requires_grad = False

        # Optimizer
        self.nbs = 64  # nominal batch size
        self.accumulate = max(round(self.nbs / self.total_batch_size), 1)  # accumulate loss before optimizing
        self.hyp['weight_decay'] *= self.total_batch_size * self.accumulate / self.nbs  # scale weight_decay
        self.logger.info(f"Scaled weight_decay = {self.hyp['weight_decay']}")

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in self.model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay

        if self.opt.adam:
            self.optimizer = optim.Adam(pg0, lr=self.hyp['lr0'],
                                        betas=(self.hyp['momentum'], 0.999))  # adjust beta1 to momentum
        else:
            self.optimizer = optim.SGD(pg0, lr=self.hyp['lr0'], momentum=self.hyp['momentum'], nesterov=True)

        self.optimizer.add_param_group(
            {'params': pg1, 'weight_decay': self.hyp['weight_decay']})  # add pg1 with weight_decay
        self.optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        self.logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2

        # resume
        if self.pretrained:
            # Optimizer
            if ckpt['optimizer'] is not None:
                self.optimizer.load_state_dict(ckpt['optimizer'])
                self.best_fitness = ckpt['best_fitness']

            # Results
            if ckpt.get('training_results') is not None:
                with open(self.results_file, 'w') as file:
                    file.write(ckpt['training_results'])  # write results.txt

            # Epochs
            self.start_epoch = ckpt['epoch'] + 1
            if self.opt.resume:
                assert self.start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' \
                                             % (self.weights, self.epochs)
            if self.epochs < self.start_epoch:
                self.logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                                 (self.weights, ckpt['epoch'], self.epochs))
                self.epochs += ckpt['epoch']  # finetune additional epochs

            del ckpt, state_dict

        # Image sizes
        self.gs = int(self.model.stride.max())  # grid size (max stride)
        self.nl = self.model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
        self.imgsz, self.imgsz_test = [check_img_size(x, self.gs) for x in
                                       self.opt.img_size]  # verify imgsz are gs-multiples

        # DP mode
        if self.cuda and self.rank == -1 and torch.cuda.device_count() > 1:
            self.model = DP(self.model)

        # SyncBatchNorm
        if self.opt.sync_bn and self.cuda and self.rank != -1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
            self.logger.info('Using SyncBatchNorm()')

        # EMA Exponential Moving Average
        self.ema = ModelEMA(self.model) if self.rank in [-1, 0] else None

        # DDP mode
        if self.cuda and self.rank != -1:
            self.model = DDP(self.model, device_ids=[self.opt.local_rank], output_device=self.opt.local_rank)

    def initilize_lr_scheduler(self):
        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
        if self.opt.linear_lr:
            self.lf = lambda x: (1 - x / (self.epochs - 1)) * (1.0 - self.hyp['lrf']) + self.hyp['lrf']  # linear
        else:
            self.lf = one_cycle(1, self.hyp['lrf'], self.epochs)  # cosine 1->hyp['lrf']
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        # plot_lr_scheduler(optimizer, scheduler, epochs)

    def initialize_dataloader(self):
        self.train_path_l = self.data_dict['train']
        self.train_path_u = self.data_dict['train_u']
        self.val_path = self.data_dict['val']
        self.test_path = self.data_dict['test']
        self.nc = 1 if self.opt.single_cls else int(self.data_dict['nc'])  # number of classes
        self.names = ['item'] if self.opt.single_cls and len(self.data_dict['names']) \
                                 != 1 else self.data_dict['names']  # class names
        assert len(self.names) == self.nc, '%g names found for nc=%g dataset in %s' % \
                                           (len(self.names), self.nc, self.opt.data)  # check

        # data_loader TODO: change dataloader
        # Train_loader for label data consist of weak augment and strong augment
        # TODO: test dataloader
        self.dataloader_l, self.dataset_l, self.dataloader_u = create_dataloader_trian_mt(self.train_path_l,
                                                                                          self.train_path_u,
                                                                                          self.imgsz,
                                                                                          self.batch_size,
                                                                                          self.gs,
                                                                                          self.opt,
                                                                                          self.hyp,
                                                                                          augment=True,
                                                                                          cache=self.opt.cache_images,
                                                                                          rect=self.opt.rect,
                                                                                          rank=self.rank,
                                                                                          world_size=self.opt.world_size,
                                                                                          workers=self.opt.workers,
                                                                                          image_weights=self.opt.image_weights,
                                                                                          quad=self.opt.quad,
                                                                                          prefix=colorstr('train: ')
                                                                                          )
        self.mlc = np.concatenate(self.dataset_l.labels, 0)[:, 0].max()  # max label class
        self.nb = len(self.dataloader_l)
        assert self.mlc < self.nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (
            self.mlc, self.nc, self.opt.data, self.nc - 1)
        if self.rank in [-1, 0]:
            self.ema.updates = self.start_epoch * self.nb // self.accumulate
            self.testloader = create_dataloader(self.test_path, self.imgsz_test, self.batch_size * 2, self.gs, self.opt,
                                                hyp=self.hyp, cache=self.opt.cache_images and not self.opt.notest,
                                                rect=True,
                                                rank=-1, world_size=self.opt.world_size, workers=self.opt.workers,
                                                pad=0.5, prefix=colorstr('val: '))[0]

            if not self.opt.resume:
                labels = np.concatenate(self.dataset_l.labels, 0)
                c = torch.tensor(labels[:, 0])  # classes
                # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
                # model._initialize_biases(cf.to(device))
                if self.plots:
                    plot_labels(labels, self.save_dir)
                    if self.tb_writer:
                        self.tb_writer.add_histogram('classes', c, 0)

                # Anchors
                if not self.opt.noautoanchor:
                    check_anchors(self.dataset_l, model=self.model, thr=self.hyp['anchor_t'], imgsz=self.imgsz)

        # Model parameters
        self.hyp['box'] *= 3. / self.nl  # scale to layers
        self.hyp['cls'] *= self.nc / 80. * 3. / self.nl  # scale to classes and layers
        self.hyp['obj'] *= (self.imgsz / 640) ** 2 * 3. / self.nl  # scale to image size and layers
        # for student model
        self.model.nc = self.nc  # attach number of classes to model
        self.model.hyp = self.hyp  # attach hyperparameters to model
        self.model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
        # attach class weights
        self.model.class_weights = labels_to_class_weights(self.dataset_l.labels, self.nc).to(
            self.device) * self.nc
        self.model.names = self.names

        # for teacher model
        self.teacher_model.nc = self.nc  # attach number of classes to model
        self.teacher_model.hyp = self.hyp  # attach hyperparameters to model
        self.teacher_model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
        self.teacher_model.class_weights = labels_to_class_weights(self.dataset_l.labels, self.nc).to(
            self.device) * self.nc  # attach class weights
        self.teacher_model.names = self.names
        # # setting to eval mode
        self.teacher_model.eval()

        # nb = len(self.dataloader_u)
        # pbar_u = enumerate(self.dataloader_u)
        # if self.rank in [-1, 0]:
        #     pbar_u = tqdm(pbar_u, total=nb)
        # for i, (imgs_weak, imgs_str, paths) in pbar_u:
        #     for path in paths:
        #         print(path)

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.996):
        # if self.rank in [-1, 0]:
        #     student_model_dict = self.ema.ema.state_dict()
        # else:
        #     student_model_dict = self.model.module.state_dict() if is_parallel(self.model) else self.model.state_dict()
        student_model_dict = self.model.module.state_dict() if is_parallel(self.model) else self.model.state_dict()
        # student_model_dict = self.ema.ema.state_dict()
        new_teacher_dict = OrderedDict()
        for key, value in self.teacher_model.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                        student_model_dict[key] * (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.teacher_model.load_state_dict(new_teacher_dict)
