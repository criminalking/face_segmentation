#!/usr/bin/env python
import numpy as np
from PIL import Image
from PIL import ExifTags
import matplotlib.pyplot as plt
from scipy import ndimage
import os

import caffe
import argparse

from face_alignment.api import FaceAlignment, LandmarksType, NetworkSize
from util import *

def main(args):
    image_paths = read_list(args.image_list)

    # init
    caffe.set_device(0)
    caffe.set_mode_gpu()

    # load net
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)

    for path in image_paths:
        # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe

        if path[-3:] == 'jpg' or path[-3:] == 'png':
            imi = open_image(path)
            # resize for memory
            width, height = imi.size
            if height > 800:
                imi = imi.resize((int(800*width/height), 800))
        else:
            continue

        # use 2D-FAN detect landmarks
        fa = FaceAlignment(LandmarksType._2D, enable_cuda=True,
                           flip_input=False, use_cnn_face_detector=True)
        try:
            landmarks = fa.get_landmarks(np.array(imi))[-1]
            landmarks = landmarks.astype('uint16')
        except:
            continue

        if args.crop == 'middle':
            imi, landmarks = crop_image_middle(landmarks, imi)
        elif args.crop == 'min':
            imi, landmarks = crop_image_min(landmarks, imi)

        if '300' in args.prototxt:
            imi = imi.resize((300, 300))
        else:
            imi = imi.resize((500, 500))
        im = np.array(imi, dtype=np.float32)
        im = im[:,:,::-1]
        im -= np.array((104.00698793,116.66876762,122.67891434))
        im = im.transpose((2,0,1))

        # shape for input (data blob is N x C x H x W), set data
        net.blobs['data'].reshape(1, *im.shape)
        net.blobs['data'].data[...] = im

        # run net and take argmax for prediction
        net.forward()
        mask = net.blobs['score'].data[0].argmax(axis=0)
        im_seg = imi * np.tile((mask!=0)[:,:,np.newaxis], (1,1,3))

        save = True if args.save == 'True' else False
        path = path[:-1] if path[-1] == '/' else path
        if '300' in args.prototxt:
            image_name = path[path.rindex('/')+1:-4] + '_yuval_300_nocrf_' + args.crop + '.png'
        else:
            image_name = path[path.rindex('/')+1:-4] + '_yuval_nocrf_' + args.crop + '.png'

        show_result(imi, mask, im_seg, save=save, filename='images/'+image_name)

        # generate prob
        #prob = np.concatenate(((1-mask)[np.newaxis,:,:]*0.9 + mask[np.newaxis,:,:]*0.1, mask[np.newaxis,:,:]*0.9+(1-mask)[np.newaxis,:,:]*0.1), axis=0)
        prob = ndimage.gaussian_filter(mask*1.0, sigma=5)
        prob = np.concatenate(((1-prob)[np.newaxis,:,:], prob[np.newaxis,:,:]), axis=0)

        # add CRF
        map = CRF(prob, np.array(imi))
        if '300' in args.prototxt:
            image_name = path[path.rindex('/')+1:-4] + '_yuval_300_crf_' + args.crop + '.png'
        else:
            image_name = path[path.rindex('/')+1:-4] + '_yuval_crf_' + args.crop + '.png'
        show_result(imi, map, np.tile((map!=0)[:,:,np.newaxis], (1,1,3)) * imi, save=save, filename='images/'+image_name)
        #show_result(imi, map, np.tile((map!=0)[:,:,np.newaxis], (1,1,3)) * imi, save=True, filename=image_name)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Face segmentation of yuval.')
    parser.add_argument('--image_list', default='input/list.txt',
                        type=str, help='path to image')
    parser.add_argument('--prototxt',
                        default='model/face_seg_fcn8s_300_deploy.prototxt',
                        type=str, help='path to prototxt')
    parser.add_argument('--caffemodel',
                        default='model/face_seg_fcn8s_300.caffemodel',
                        type=str, help='path to caffemodel')
    parser.add_argument('--crop', choices=['min', 'middle', 'no'],
                        default='min', help='choose min/middle/no crop')
    parser.add_argument('--save', choices=['True', 'False'],
                        default='False', help='choose if save final result')
    args = parser.parse_args()

    main(args)
