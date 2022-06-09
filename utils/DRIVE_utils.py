import cv2
import numpy as np
import torch
import torch.utils.data as data

from utils.aug_utils import (randomHorizontalFlip, randomHueSaturationValue, randomRotate90, 
                            randomShiftScaleRotate, randomVerticalFlip)
from PIL import Image
import os


def default_DRIVE_loader(cfg, img_path, mask_path, image_size, mode='train'):
    img = cv2.imread(img_path)
    img = cv2.resize(img, image_size)
    original_img = img
    # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = np.array(Image.open(mask_path))

    mask = cv2.resize(mask, image_size)
    if mode.lower()=='train' or mode.lower()=='val':
        if cfg.AUGMENT.HUE_SATURATION_VALUE:    img = randomHueSaturationValue(img,
                                                hue_shift_limit=(-30, 30),
                                                sat_shift_limit=(-5, 5),
                                                val_shift_limit=(-15, 15), u=cfg.AUGMENT.HUE_SATURATION_VALUE_PROBABILITY)
        if cfg.AUGMENT.SHIFT_SCALE_ROTATE: img, mask = randomShiftScaleRotate(img, mask,
                                           shift_limit=(-0.1, 0.1),
                                           scale_limit=(-0.1, 0.1),
                                           aspect_limit=(-0.1, 0.1),
                                           rotate_limit=(-0, 0), u=cfg.AUGMENT.SHIFT_SCALE_ROTATE_PROBABILITY)

        if cfg.AUGMENT.HORIZONTAL_FLIP: img, mask = randomHorizontalFlip(img, mask, u=cfg.AUGMENT.HORIZONTAL_FLIP_PROBABILITY)
        if cfg.AUGMENT.VERTICAL_FLIP: img, mask = randomVerticalFlip(img, mask, u=cfg.AUGMENT.VERTICAL_FLIP_PROBABILITY)
        if cfg.AUGMENT.ROTATE: img, mask = randomRotate90(img, mask, u=cfg.AUGMENT.ROTATE_PROBABILITY)


    mask = np.expand_dims(mask, axis=2)
    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    # mask = abs(mask-1)
    return img, mask, original_img



def read_DRIVE_datasets(root_path, mode):
    images = []
    masks = []

    if(mode.lower() == 'train'):
      image_root = os.path.join(root_path, 'training/images')
      gt_root = os.path.join(root_path, 'training/1st_manual')
    
    elif(mode.lower() in ['val', 'test', 'testval']):
      image_root = os.path.join(root_path, 'test/images')
      gt_root = os.path.join(root_path, 'test/1st_manual')

    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name.split('.')[0] + '.tif')
        label_path = os.path.join(gt_root, image_name.split('_')[0] + '_manual1.gif')

        images.append(image_path)
        masks.append(label_path)
    
    if(mode.lower() == 'val'):
        images = images[0:10]
        masks = masks[0:10]
    elif(mode.lower() == 'test'):
        images = images[10:20]
        masks = masks[10:20]

    print(images)
    print(masks)
    return images, masks



class ImageFolder(data.Dataset):

    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        self.mode = mode
        self.images, self.labels = read_DRIVE_datasets(cfg.DATASET.DATASET_DIR, self.mode)
        self.image_size = cfg.TEST.IMAGE_SIZE if mode=='test' else cfg.DATASET.IMAGE_SIZE

    def __getitem__(self, index):
        img, mask, original_img = default_DRIVE_loader(self.cfg, self.images[index], self.labels[index], self.image_size, self.mode)
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return img, mask, original_img

    def __len__(self):
        assert len(self.images) == len(self.labels), 'The number of images must be equal to labels'
        return len(self.images)

