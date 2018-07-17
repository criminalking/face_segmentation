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

# Comment this out to run the code on the CPU
caffe.set_mode_gpu()
caffe.set_device(0)

# Load both networks
net1 = caffe.Net('model/net_landmarks.prototxt', \
                 'model/params_landmarks.caffemodel', caffe.TEST)
net2 = caffe.Net('model/net_segmentation.prototxt', \
                 'model/params_segmentation.caffemodel', caffe.TEST)

# Generate the colourmap for the segmentation mask
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

# We have a Gaussian to recover the output slightly - better results
f = scipy.io.loadmat('gaus.mat')['f']

# load image names
imNames = [ line.rstrip() for line in listdir('input/') ]

# segment and measure performance
for imIdx, imName in enumerate(imNames):
    if imName[-3:] == 'jpg':
        imName = imName[:-4]
        imi = Image.open('input/' + imName + '.jpg')
    elif imName[-3:] == 'png':
        imName = imName[:-4]
        imi = Image.open('input/' + imName + '.png')
    print imName

    # prepare the image, limit image size for memory
    height, width = imi.size
    if height > width:
        if height > 450:
            imi = imi.resize((450, int(450 * width/height)))
        #elif width < 300:
        #    imi = imi.resize((int(300 * height/width), 300))
    elif width >= height:
        if width > 450:
            imi = imi.resize((int(450 * height/width), 450))
        #elif height < 300:
        #    imi = imi.resize((300, int(300 * width/height)))
    height, width = imi.size
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
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(imi)
    ax1.scatter(Q[:,1], Q[:,0], c='r', s=20)
    ax2.imshow(S)
    #plt.show()

    # transfer score to probability with softmax for later unary term
    score = net2.blobs['score'].data[0]

    prob = np.exp(score) / np.sum(np.exp(score), 0)
    prob_max = np.max(prob, 0) # (0.28, 1)

    # CRF
    import pydensecrf.densecrf as dcrf
    nlabels = 1
    d = dcrf.DenseCRF2D(width, height, nlabels)
    # set Unary
    U = -np.log(prob_max+1e-6)
    U = U.astype('float32')
    U = U.reshape(nlabels, -1) # needs to be flat
    U = np.ascontiguousarray(U)
    d.setUnaryEnergy(U)
    # set Pairwise
    im = im.transpose((1,0,2)) # width, height
    im = np.ascontiguousarray(im).astype('uint8')
    d.addPairwiseGaussian(sxy=(3,3), compat=3)
    d.addPairwiseBilateral(sxy=(80,80), srgb=(13,13,13), rgbim=im, compat=10)
    Q = d.inference(5)
    map = np.argmax(Q, axis=0).reshape((height,width))
    print np.sum(map!=0)
    plt.imshow(map.transpose((1,0)))
    plt.show()

    exit(0)
