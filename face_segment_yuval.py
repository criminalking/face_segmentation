#!/usr/bin/env python
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import caffe
import argparse


def main(args):
    # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    im = Image.open(args.input)
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
    out = net.blobs['score'].data[0].argmax(axis=0)

    im_seg = im * np.tile((out!=0)[:,:,np.newaxis], (1,1,3))
    plt.imsave('seg.png', im_seg)

    plt.imshow(out)
    plt.imsave('output.png', out)
    plt.draw()
    plt.pause(0.001)
    plt.waitforbuttonpress()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=
                                     'Face segmentation of yuval.')
    parser.add_argument('--input',
        default='input/000000_left.png',
        type=str, help='path to image')
    parser.add_argument('--prototxt',
        default='model/face_seg_fcn8s_deploy.prototxt',
        type=str, help='path to prototxt')
    parser.add_argument('--caffemodel',
        default='model/face_seg_fcn8s.caffemodel',
        type=str, help='path to caffemodel')
    args = parser.parse_args()

    main(args)
