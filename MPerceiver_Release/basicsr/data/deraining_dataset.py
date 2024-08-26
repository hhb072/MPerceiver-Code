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

@DATASET_REGISTRY.register(suffix='basicsr')
class deraining_dataset(data.Dataset):
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
        super(deraining_dataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        if 'crop_size' in opt:
            self.crop_size = opt['crop_size']
        else:
            self.crop_size = 512
        if 'image_type' not in opt:
            opt['image_type'] = 'png'

        # support multiple type of data: file path and meta data, remove support of lmdb
        self.paths_gt = []
        self.paths_lq = []

        if 'gt_path' in opt:
            if isinstance(opt['gt_path'], str):
                self.paths_gt.extend(sorted([str(x) for x in Path(opt['gt_path']).glob('*.'+opt['image_type'])]))
            else:
                self.paths_gt.extend(sorted([str(x) for x in Path(opt['gt_path'][0]).glob('*.'+opt['image_type'])]))
                if len(opt['gt_path']) > 1:
                    for i in range(len(opt['gt_path'])-1):
                        self.paths_gt.extend(sorted([str(x) for x in Path(opt['gt_path'][i+1]).glob('*.'+opt['image_type'])]))
        
        if 'lq_path' in opt:
            if isinstance(opt['lq_path'], str):
                self.paths_lq.extend(sorted([str(x) for x in Path(opt['lq_path']).glob('*.'+opt['image_type'])]))
            else:
                self.paths_lq.extend(sorted([str(x) for x in Path(opt['lq_path'][0]).glob('*.'+opt['image_type'])]))
                if len(opt['lq_path']) > 1:
                    for i in range(len(opt['lq_path'])-1):
                        self.paths_lq.extend(sorted([str(x) for x in Path(opt['lq_path'][i+1]).glob('*.'+opt['image_type'])]))

        # # limit number of pictures for test
        # if 'num_pic' in opt:
        #     if 'val' or 'test' in opt:
        #         random.shuffle(self.paths)
        #         self.paths = self.paths[:opt['num_pic']]
        #     else:
        #         self.paths = self.paths[:opt['num_pic']]

        # if 'mul_num' in opt:
        #     self.paths = self.paths * opt['mul_num']
        #     # print('>>>>>>>>>>>>>>>>>>>>>')
        #     # print(self.paths)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths_gt[index]
        lq_path = self.paths_lq[index]
        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                img_bytes_gt = self.file_client.get(gt_path, 'gt')
                img_bytes_lq = self.file_client.get(lq_path, 'lq')

            except (IOError, OSError) as e:
                # logger = get_root_logger()
                # logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__()-1)
                gt_path = self.paths_gt[index]
                lq_path = self.paths_lq[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        img_gt = imfrombytes(img_bytes_gt, float32=True)
        img_lq = imfrombytes(img_bytes_lq, float32=True)

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_gt,img_lq = augment([img_gt,img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # crop or pad to 400
        # TODO: 400 is hard-coded. You may change it accordingly
        h, w = img_gt.shape[0:2]
        crop_pad_size = self.crop_size
        # pad
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            img_lq = cv2.copyMakeBorder(img_lq, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        # crop
        if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
            h, w = img_gt.shape[0:2]
            # randomly choose top and left coordinates
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)
            # top = (h - crop_pad_size) // 2 -1
            # left = (w - crop_pad_size) // 2 -1
            img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]
            img_lq = img_lq[top:top + crop_pad_size, left:left + crop_pad_size, ...]

       

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        img_lq = img2tensor([img_lq], bgr2rgb=True, float32=True)[0]

        return_d = {'gt': img_gt, 'gt_path': gt_path, 'lq':img_lq, 'lq_path': lq_path}
        return return_d

    def __len__(self):
        return len(self.paths_gt)
