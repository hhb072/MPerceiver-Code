import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from pathlib import Path
from torch.utils import data as data

from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from PIL import Image

# @DATASET_REGISTRY.register(suffix='basicsr')
class one_dataset(data.Dataset):
    """Modified dataset based on the dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, opt):
        super(one_dataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        if 'crop_size' in opt:
            self.crop_size = opt['crop_size']
        else:
            self.crop_size = 512
        if 'image_type' not in opt:
            opt['image_type'] = 'png'



        in_files = os.listdir(opt["root_lq"])

        self.imgs_in=[os.path.join(opt["root_lq"], k) for k in in_files]
        self.imgs_gt=[os.path.join(opt["root_gt"], k) for k in in_files]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)    
        # avoid errors caused by high latency in reading files

        retry = 3
        gt_path = self.imgs_gt[index]
        lq_path = self.imgs_in[index]
        while retry > 0:
                try:
                    img_bytes_gt = self.file_client.get(gt_path, 'gt')
                    img_bytes_lq = self.file_client.get(lq_path, 'lq')
                except (IOError, OSError) as e:
                    # logger = get_root_logger()
                    # logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                    # change another file to read
                    index = random.randint(0, self.__len__()-1)
                    gt_path = self.imgs_gt[index]
                    lq_path = self.imgs_in[index]
                    time.sleep(1)  # sleep 1s for occasional server congestion
                else:
                    break
                finally:
                    retry -= 1
        img_gt = imfrombytes(img_bytes_gt, float32=True)
        img_lq = imfrombytes(img_bytes_lq, float32=True)
        
        # gt_caption = Image.open(gt_path).convert('RGB') 
        # gt_caption = np.array(gt_caption)

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_gt,img_lq = augment([img_gt,img_lq], self.opt['use_hflip'], self.opt['use_rot'])
        # img_gts = augment(img_gts, self.opt['use_hflip'], self.opt['use_rot'])
        # img_lqs = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        # crop or pad to 400
        # TODO: 400 is hard-coded. You may change it accordingly         
        h, w = img_gt.shape[0:2]
        crop_pad_size = self.crop_size
            # pad
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt= cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            img_lq = cv2.copyMakeBorder(img_lq, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
                # gt_caption = cv2.copyMakeBorder(gt_caption, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

            # crop
        if img_gt.shape[0] >= crop_pad_size or img_gt.shape[1] >= crop_pad_size:
            h, w = img_gt.shape[0:2]
                # randomly choose top and left coordinates
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)
                # top = (h - crop_pad_size) // 2 -1
                # left = (w - crop_pad_size) // 2 -1
            img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]
            img_lq = img_lq[top:top + crop_pad_size, left:left + crop_pad_size, ...]
                # gt_caption = gt_caption[top:top + crop_pad_size, left:left + crop_pad_size, ...]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)
        return_d = {'gt': img_gt, 'lq':img_lq}
        return return_d

    def __len__(self):
        return len(self.imgs_in)
    
