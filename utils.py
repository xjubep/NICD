import random

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_optimizer as optim
from torch.optim.lr_scheduler import _LRScheduler

from models.frame import MobileNetV3Small, EfficientNetb3, DOLG, DOLG_FF
from models.triplet import TripletNet


def seed_everything(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args):
    model = None
    if args.model == 'mobilenet':
        model = MobileNetV3Small()
    elif args.model == 'efficientnet':
        model = EfficientNetb3()
    # elif args.model == 'hybrid_vit':
    #     model = HybridViT()
    elif args.model == 'mobilenet_dolg':
        model = DOLG(arch='mobilenetv3_small_100')
    elif args.model == 'mobilenet_dolg_df':
        model = DOLG(arch='mobilenetv3_small_100')
    elif args.model == 'mobilenet_dolg_ff':
        model = DOLG_FF(arch='mobilenetv3_small_100')
    model = model.to(args.device)

    # for name, param in model.named_parameters():
    #     if 'backbone' in name.split('.'):
    #         param.requires_grad = False

    triple_model = TripletNet(model)
    triple_model = triple_model.to(args.device)

    if torch.cuda.device_count() > 1:
        triple_model = torch.nn.DataParallel(triple_model)
    return triple_model


def build_optimizer(args, model):
    optimizer = None
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=args.learning_rate,
                                     weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=args.learning_rate,
                                      weight_decay=args.weight_decay)
    elif args.optimizer == 'radam':
        optimizer = optim.RAdam(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)
    elif args.optimizer == 'adamp':
        optimizer = optim.AdamP(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)
    elif args.optimizer == 'ranger':
        optimizer = optim.Ranger(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.learning_rate,
                                 betas=(0.9, 0.999),
                                 weight_decay=args.weight_decay)
    elif args.optimizer == 'lamb':
        optimizer = optim.Lamb(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.learning_rate,
                               weight_decay=args.weight_decay)
    elif args.optimizer == 'adabound':
        optimizer = optim.AdaBound(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=args.learning_rate,
                                   weight_decay=args.weight_decay)
    return optimizer


def build_loss(args):
    criterion = None
    if args.distance == 'l2':
        criterion = nn.TripletMarginWithDistanceLoss(margin=args.margin, distance_function=nn.PairwiseDistance())
    elif args.distance == 'cos':
        criterion = nn.TripletMarginWithDistanceLoss(margin=args.margin,
                                                     distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y))

    return criterion


def build_scheduler(args, optimizer, iter_per_epoch):
    scheduler = None
    if args.scheduler == 'cos_base':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'cos':
        # tmax = epoch * 2 => half-cycle
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tmax, eta_min=args.min_lr)
    elif args.scheduler == 'cycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr,
                                                        steps_per_epoch=iter_per_epoch, epochs=args.epochs)

    return scheduler


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def load_descriptor_h5(descs_submission_path):
    """Load datasets from descriptors submission hdf5 file."""

    with h5py.File(descs_submission_path, "r") as f:
        query = f["query"][:]
        reference = f["reference"][:]
        # Coerce IDs to native Python unicode string no matter what type they were before
        query_ids = np.array(f["query_ids"][:], dtype=object).astype(str).tolist()
        reference_ids = np.array(f["reference_ids"][:], dtype=object).astype(str).tolist()

        if "train" in f:
            train = f["train"][:]
        else:
            train = None

    return query, reference, train, query_ids, reference_ids
