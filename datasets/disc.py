import os
import random
from glob import glob

import albumentations as A
import cv2
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset


class DISC(Dataset):
    '''
        DISC
            images
                dev_queries
                    Qaaaaa.jpg
                references
                    Rbbbbbb.jpg
                train
                    Tcccccc.jpg
        Train: Triplet
        Test: Two image -> one by one
    '''

    def __init__(self, args, img_root, num_negatives=None, triplet_csv=None, train_csv=None, test_csv=None,
                 query_csv=None, ref_csv=None, mode='train_triplet'):
        self.args = args
        self.img_root = img_root
        self.img_size = args.img_size
        self.mode = mode

        if self.mode == 'train_triplet':
            self.triplets = pd.read_csv(triplet_csv).values

            self.transform = A.Compose([
                A.RandomResizedCrop(height=self.img_size, width=self.img_size,
                                    scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333),
                                    interpolation=1, p=1.0),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30,
                                   interpolation=1, border_mode=0, value=0, p=0.5),
                A.OneOf([
                    A.HorizontalFlip(p=1.0),
                    A.RandomRotate90(p=1.0),
                    A.VerticalFlip(p=1.0)
                ], p=0.5),
                A.OneOf([
                    A.CLAHE(clip_limit=2, p=1.0),
                    A.Sharpen(p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=1.0),
                ], p=0.25),
                A.OneOf([
                    A.MotionBlur(p=1),
                    A.OpticalDistortion(p=1),
                    A.GaussNoise(p=1)
                ], p=0.25),
                A.ImageCompression(quality_lower=80, quality_upper=100, p=0.1),
                A.ToGray(p=0.05),
                A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2()
            ])
        elif self.mode == 'train_online':
            self.train_paths = pd.read_csv(train_csv).values
            self.neg_paths = glob(f'{self.img_root}/train/*')
            self.num_negatives = num_negatives

            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.OneOf([
                    A.HorizontalFlip(p=1.0),
                    A.RandomRotate90(p=1.0),
                    A.VerticalFlip(p=1.0)
                ], p=0.25),
                A.ImageCompression(quality_lower=80, quality_upper=100, p=0.1),
                A.ToGray(p=0.05),
                A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2()
            ])
        elif self.mode == 'query':
            self.query_paths = pd.read_csv(query_csv).values

            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2()
            ])
        elif self.mode == 'ref':
            self.ref_paths = pd.read_csv(ref_csv).values

            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                ToTensorV2()
            ])

    def load_image(self, path):
        img_path = os.path.join(self.img_root, path)
        img = cv2.imread(f'{img_path}')
        img = self.transform(image=img)
        return img_path, img['image']

    def load_image_flip(self, path):
        img_path = os.path.join(self.img_root, path)
        img = cv2.imread(f'{img_path}')
        img1 = self.transform(image=img)
        img = cv2.flip(img, 1)
        img2 = self.transform(image=img)
        return img_path, img1['image'], img2['image']

    def __len__(self):
        if self.mode == 'train_triplet':
            return len(self.triplets)
        elif self.mode == 'train_online':
            return len(self.train_paths)
        elif self.mode == 'query':
            return len(self.query_paths)
        elif self.mode == 'ref':
            return len(self.ref_paths)

    def __getitem__(self, idx):
        if self.mode == 'train_triplet':
            a, p, n = self.triplets[idx]

            anc_path, anc = self.load_image(a)
            pos_path, pos = self.load_image(p)
            neg_path, neg = self.load_image(n)

            return {'anc_path': anc_path, 'pos_path': pos_path, 'neg_path': neg_path,
                    'anc': anc, 'pos': pos, 'neg': neg}
        elif self.mode == 'train_online':
            a, p = self.train_paths[idx]
            ns = [random.choice(range(len(self.neg_paths))) for _ in range(self.num_negatives)]

            anc_path, anc = self.load_image(a)
            pos_path, pos = self.load_image(p)

            ret = [
                anc,
                pos,
                *[self.transform(image=cv2.imread(f'{self.neg_paths[n]}'))['image'] for n in ns]]
            return idx, ret, torch.tensor([n + 1_000_000 for n in ns])
        elif self.mode == 'query':
            q = self.query_paths[idx][0]
            path, img, img_flip = self.load_image_flip(q)
            return {'path': path, 'img': img, 'img_flip': img_flip}
        elif self.mode == 'ref':
            r = self.ref_paths[idx][0]
            path, img, img_flip = self.load_image_flip(r)
            return {'path': path, 'img': img, 'img_flip': img_flip}
