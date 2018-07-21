#!/usr/bin/env python
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import caffe
import argparse

from util import *


def main(args):
    image_paths = read_list(args.image_list)

    for path in image_paths:
        # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
        im = Image.open(path)

        if args.crop != 'no':
            # use landmarks to crop image
            landmark_file = path[:-3] + 'txt'
            landmarks = load_landmarks(landmark_file)
            if args.crop == 'middle':
                im = crop_image_middle(landmarks, im)
            else:
                im = crop_image_min(landmarks, im)
                args.crop = 'min'

        if '300' in args.prototxt:
            im = im.resize((300, 300))
        else:
            im = im.resize((500, 500))
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= np.array((104.00698793,116.66876762,122.67891434))
        in_ = in_.transpose((2,0,1))

        # init
        caffe.set_device(0)
        caffe.set_mode_gpu()

        # load net
        net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)

        # shape for input (data blob is N x C x H x W), set data
        net.blobs['data'].reshape(1, *in_.shape)
        net.blobs['data'].data[...] = in_

        # run net and take argmax for prediction
        net.forward()
        mask = net.blobs['score'].data[0].argmax(axis=0)

        im_seg = im * np.tile((mask!=0)[:,:,np.newaxis], (1,1,3))

        path = path[:-1] if path[-1] == '/' else path

        if '300' in args.prototxt:
            image_name = path[path.rindex('/')+1:-4] + '_yuval_300_nocrf_' + args.crop + '.png'
        else:
            image_name = path[path.rindex('/')+1:-4] + '_yuval_nocrf_' + args.crop + '.png'
        show_result(im, mask, im_seg, save=True, filename='images/'+image_name)

        # add CRF
        prob = np.concatenate(((1-mask)[np.newaxis,:,:]*0.9 + mask[np.newaxis,:,:]*0.1, mask[np.newaxis,:,:]*0.9+(1-mask)[np.newaxis,:,:]*0.1), axis=0)
        map = CRF(prob, np.array(im))
        if '300' in args.prototxt:
            image_name = path[path.rindex('/')+1:-4] + '_yuval_300_crf_' + args.crop + '.png'
        else:
            image_name = path[path.rindex('/')+1:-4] + '_yuval_crf_' + args.crop + '.png'
        show_result(im, map, np.tile((map!=0)[:,:,np.newaxis], (1,1,3)) * im, save=True, filename='images/'+image_name)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Face segmentation of yuval.')
    parser.add_argument('--image_list', default='data/images/list.txt',
                        type=str, help='path to image')
    parser.add_argument('--prototxt',
                        default='data/face_seg_fcn8s_300_deploy.prototxt',
                        type=str, help='path to prototxt')
    parser.add_argument('--caffemodel',
                        default='data/face_seg_fcn8s_300.caffemodel',
                        type=str, help='path to caffemodel')
    parser.add_argument('--crop', choices=['min', 'middle', 'no'],
                        help='choose min/middle/no crop')
    args = parser.parse_args()

    main(args)
