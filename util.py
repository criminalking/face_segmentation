import numpy as np
import matplotlib.pyplot as plt

import pydensecrf.densecrf as dcrf

def read_list(filename):
    img_list = []
    with open(filename, 'r') as f:
        for line in f.readlines()[0:]:
            if line[0] == '#':
                continue
            img_path = line.strip().split()
            img_list.append(img_path[0])
    return img_list


def crop_image(image):
    cropped_image = image
    return cropped_image


def load_landmarks(filename, number=68):
    """Load landmarks

    Note:
        Format:
        x1,y1
        x2,y2

    """
    landmarks = np.zeros((number, 2))
    try:
        with open(filename, 'r') as f:
            data = f.read()
    except OSError:
        print('cannot open', filename)

    lines = data.splitlines()
    for index, line in enumerate(lines):
        if line == '':
            continue
        elem = line.split(',')
        landmarks[index][0] = float(elem[0])
        landmarks[index][1] = float(elem[1])
    return landmarks


def show_result(image, mask, seg, save=False):
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.imshow(image)
    ax2.imshow(mask)
    ax3.imshow(seg)
    plt.show()

    if save:
        plt.imsave('seg.png', seg)
        plt.imsave('mask.png', mask)


def CRF(prob, im):
    height, width, _ = im.shape
    nlabels = prob.shape[0]
    d = dcrf.DenseCRF2D(width, height, nlabels)
    # set Unary
    U = -np.log(prob+1e-6)
    U = U.astype('float32')
    U = U.reshape(nlabels, -1) # needs to be flat
    U = np.ascontiguousarray(U)
    d.setUnaryEnergy(U)
    # set Pairwise
    im = np.ascontiguousarray(im).astype('uint8')
    d.addPairwiseGaussian(sxy=(3,3), compat=3)
    d.addPairwiseBilateral(sxy=(50,50), srgb=(20,20,20), rgbim=im, compat=10)
    Q = d.inference(20)
    map = np.argmax(Q, axis=0).reshape((height,width))
    return map
