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

from util import read_list, CRF, show_result

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
    net1 = caffe.Net('model/net_landmarks.prototxt', \
                     'model/params_landmarks.caffemodel', caffe.TEST)
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
        print imName

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
        lanIm = im - np.array((122.67,104.00,116.67))
        segIm = im - np.array((87.86,101.92,133.01))
        lanIm = lanIm.transpose((2,0,1))
        segIm = segIm.transpose((2,0,1))

        # Forward through the landmark network
        net1.blobs['data'].reshape(1, *lanIm.shape)
        net1.blobs['data'].data[0,:,:,:] = lanIm
        net1.forward()
        H = net1.blobs['score'].data[0]

        # Do some recovery of the points
        C = np.zeros(H.shape, 'uint8') # cleaned up heatmaps
        C = np.pad(C, ((0,0), (120,120), (120,120)), 'constant')
        Q = np.zeros((68,2), 'float') # detected landmarks
        for k in range(0,68):
            ij = np.unravel_index(H[k,:,:].argmax(), H[k,:,:].shape)
            Q[k,0] = ij[0]
            Q[k,1] = ij[1]
            C[k,ij[0]+120-100:ij[0]+120+101,ij[1]+120-100:ij[1]+120+101] = f
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
        show_result(imi, S, np.tile((mask!=0)[:,:,np.newaxis], (1,1,3)) * imi)
        show_result(imi, map, np.tile((map!=0)[:,:,np.newaxis], (1,1,3)) * imi)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=
                                     'Face part segmentation.')
    parser.add_argument('--image_list', default='input/list.txt', type=str,
                        help='path to input images')
    args = parser.parse_args()
    main(args)
