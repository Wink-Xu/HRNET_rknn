import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
import argparse
from rknn.api import RKNN

ONNX_MODEL = '../../output/pose_hrnet_w32_256x192.onnx'
RKNN_MODEL = '../../output/pose_hrnet_w32_256x192_notPre.rknn'

def save_image_with_joints(image, joints_list, file_name):

    for joint in joints_list[0]:
        joint[0] = joint[0] * 4
        joint[1] = joint[1] * 4    
        cv2.circle(image, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)

    ndarr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_name, ndarr)

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals
    


def parse_args():
    parser = argparse.ArgumentParser(description='rknn toolkit')
    # general
    parser.add_argument(
    '--transModel',
    action='store_true',
    help='whether to transform model')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # Create RKNN object
    rknn = RKNN()
    
    if args.transModel:
    # pre-process config
        print('--> Config model')
        rknn.config(mean_values=[[123.675, 116.28, 103.53]], std_values=[[58.40, 57.12, 57.38]], reorder_channel='0 1 2', target_platform=['rv1126'])
        print('done')

        # Load ONNX model
        print('--> Loading model')
        ret = rknn.load_onnx(model=ONNX_MODEL)
        if ret != 0:
            print('Load pose_hrnet_w32_256x192 failed!')
            exit(ret)
        print('done')

        # Build model
        print('--> Building model')
        ret = rknn.build(do_quantization=True, dataset='./dataset.txt')#, pre_compile = True)   ## pre_compile  inferece on device
        if ret != 0:
            print('Build pose_hrnet_w32_256x192 failed!')
            exit(ret)
        print('done')

        # Export RKNN model
        print('--> Export RKNN model')
        ret = rknn.export_rknn(RKNN_MODEL)
        if ret != 0:
            print('Export pose_hrnet_w32_256x192.rknn failed!')
            exit(ret)
        print('done')

        rknn.release()
    else:
        ret = rknn.load_rknn(path=RKNN_MODEL)

        # Set inputs
        img = cv2.imread('../../output/test2.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 192))

        # init runtime environment
        print('--> Init runtime environment')
        ret = rknn.init_runtime()
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        print('done')

        # Inference
        print('--> Running model')
        outputs = rknn.inference(inputs=[img])
        x = outputs[0] 
        
        print(x.flatten()[:20])

        pred, _ = get_max_preds(x)
        print(pred)
        filename = '../../output/test2_rknn_out.jpg'
        save_image_with_joints(img,  pred, filename)

        print('done')

        rknn.release()

