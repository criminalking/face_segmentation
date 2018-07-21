#!/usr/bin/env python
from __future__ import division
import sys, os
import caffe
import numpy as np
from PIL import Image
from PIL import ImageDraw
import scipy.io
import string
import matplotlib.pyplot as plt
from os import listdir
import argparse

from face_alignment.api import FaceAlignment, LandmarksType, NetworkSize
from util import *

def get_palette():
    """Generate the colourmap for the segmentation mask."""
    palette = np.zeros((255,3))
    palette[0,:] = [   0,   0,   0] # Background
    palette[1,:] = [ 255, 184, 153] # Skin
    palette[2,:] = [ 112,  65,  57] # Eyebrows
    palette[3,:] = [  51, 153, 255] # Eyes
    palette[4,:] = [ 219, 144, 101] # Nose
    palette[5,:] = [ 135,   4,   0] # Upper lip
    palette[6,:] = [  67,   0,   0] # Mouth
    palette[7,:] = [ 135,   4,   0] # Lower lip
    palette = palette.astype('uint8').tostring()

    return palette


def main(args):
    caffe.set_mode_gpu()
    caffe.set_device(0)

    # Load both networks
    #net1 = caffe.Net('model/net_landmarks.prototxt', \
    #                 'model/params_landmarks.caffemodel', caffe.TEST)
    net2 = caffe.Net('model/net_segmentation.prototxt', \
                     'model/params_segmentation.caffemodel', caffe.TEST)

    palette = get_palette()

    # We have a Gaussian to recover the output slightly - better results
    f = scipy.io.loadmat('gaus.mat')['f']

    # load image names
    image_paths = read_list(args.image_list)

    # segment and measure performance
    for imName in image_paths:
        if imName[-3:] == 'jpg' or imName[-3:] == 'png':
            imi = Image.open(imName)
        else:
            continue

        # use 2D-FAN detect landmarks
        fa = FaceAlignment(LandmarksType._2D, enable_cuda=True,
                           flip_input=False, use_cnn_face_detector=True)
        landmarks = fa.get_landmarks(np.array(imi))[-1]
        landmarks = landmarks.astype('uint16')

        if args.crop == 'middle':
            imi, landmarks = crop_image_middle(landmarks, imi)
        elif args.crop == 'min':
            imi, landmarks = crop_image_min(landmarks, imi)

        # prepare the image, limit image size for memory
        width, height = imi.size
        if width > height:
            if width > 450:
                imi = imi.resize((450, int(450 * height/width)))
            #elif height < 300:
            #    imi = imi.resize((int(300 * width/height), 300))
        else:
            if height > 450:
                imi = imi.resize((int(450 * width/height), 450))
            #elif width < 300:
            #    imi = imi.resize((300, int(300 * height/width)))
        width, height = imi.size
        im = np.array(imi, dtype=np.float32)
        if len(im.shape) == 2:
            im = np.reshape(im, im.shape+(1,))
            im = np.concatenate((im,im,im), axis=2)
        im = im[:,:,::-1] # RGB to BGR

        # trained with different means (accidently)
        segIm = im - np.array((87.86,101.92,133.01))
        segIm = segIm.transpose((2,0,1))

        # Do some recovery of the points
        C = np.zeros((landmarks.shape[0], height, width), 'uint8') # cleaned up heatmaps
        C = np.pad(C, ((0,0), (120,120), (120,120)), 'constant')

        landmarks[:,0], landmarks[:,1] = landmarks[:,1].copy(), landmarks[:,0].copy()
        for k in range(0,68):
            C[k,landmarks[k,0]+120-100:landmarks[k,0]+120+101,landmarks[k,1]+120-100:landmarks[k,1]+120+101] = f
        C = C[:,120:-120,120:-120] * 0.5

        # Forward through the segmentation network
        D = np.concatenate((segIm, C))
        net2.blobs['data'].reshape(1, *D.shape)
        net2.blobs['data'].data[0,:,:,:] = D
        net2.forward()
        mask = net2.blobs['score'].data[0].argmax(axis=0)
        S = Image.fromarray(mask.astype(np.uint8))
        S.putpalette(palette)

        print 'close figure to process next image'

        # transfer score to probability with softmax for later unary term
        score = net2.blobs['score'].data[0]
        prob = np.exp(score) / np.sum(np.exp(score), 0) # (nlabels, height, width)
        #prob_max = np.max(prob, 0) # (0.28, 1)

        # CRF
        map = CRF(prob, im) # final label

        # show result
        imName = imName[:-1] if imName[-1] == '/' else imName
        image_name = imName[imName.rindex('/')+1:-4] + '_part_nocrf_' + args.crop + '.png'
        show_result(imi, mask, np.tile((mask!=0)[:,:,np.newaxis], (1,1,3)) * imi,
                    save=True, filename='images/'+image_name)
        image_name = imName[imName.rindex('/')+1:-4] + '_part_crf_' + args.crop + '.png'
        show_result(imi, map, np.tile((map!=0)[:,:,np.newaxis], (1,1,3)) * imi,
                    save=True, filename='images/'+image_name)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=
                                     'Face part segmentation.')
    parser.add_argument('--image_list', default='input/list.txt', type=str,
                        help='path to input images')
    parser.add_argument('--crop', choices=['min', 'middle', 'no'],
                        default='min', help='choose min/middle/no crop')
    args = parser.parse_args()
    main(args)
