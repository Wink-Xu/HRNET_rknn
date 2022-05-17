# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import cv2
import numpy as np
import time 

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from config import cfg
from config import update_config
from utils.transforms import get_affine_transform
from utils.transforms import get_final_preds
from utils.transforms import get_max_preds
from utils.vis import save_debug_images
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    parser.add_argument('--imageFile',
                        help='image path',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args



def _xywh2cs(x, y, w, h):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    scale = np.array(
        [w * 1.0 / 200, h * 1.0 / 200],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def main():
    args = parse_args()
    update_config(cfg, args)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


    image_file = args.imageFile

    data_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    if cfg.DATASET.COLOR_RGB:
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

    if data_numpy is None:
        raise ValueError('Fail to read {}'.format(image_file))

    c, s = _xywh2cs(0, 0, data_numpy.shape[1], data_numpy.shape[0])
    r = 0
    
    image_size = np.array(cfg.MODEL.IMAGE_SIZE)
    trans = get_affine_transform(c, s, r, image_size)
    input = cv2.warpAffine(
        data_numpy,
        trans,
        (int(image_size[0]), int(image_size[1])),
        flags=cv2.INTER_LINEAR)

    input = transform(input)

    # switch to evaluate mode
    model.eval()

    num_samples = 1
    all_preds = np.zeros(
        (num_samples, cfg.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    idx = 0
    with torch.no_grad():
        end = time.time()
            # compute output

        input = input.reshape(1, input.shape[0], input.shape[1], input.shape[2])
        outputs = model(input)
        if isinstance(outputs, list):
            output = outputs[-1]
        else:
            output = outputs

        num_images = input.shape[0]

        end = time.time()
        
        preds, maxvals = get_final_preds(
            cfg, output.clone().cpu().numpy(), c, s)
        pred, _ = get_max_preds(output.clone().cpu().numpy())
        print(pred*4)
        all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
        all_preds[idx:idx + num_images, :, 2:3] = maxvals
     
        idx += num_images
        
        prefix = image_file[:-4]

        save_debug_images(cfg, input,  pred*4, output, 
                            prefix)


if __name__ == '__main__':
    main()
