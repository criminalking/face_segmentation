import numpy as np
import matplotlib.pyplot as plt
from PIL import ExifTags
from PIL import Image

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


def open_image(path):
    im = Image.open(path)
    if path[-3:] == 'jpg' and im._getexif():
        exif=dict((ExifTags.TAGS[k], v) for k, v in im._getexif().items() if k in ExifTags.TAGS)
        if exif['Orientation'] == 6:
            im = im.rotate(-90, expand=True)
    return im


def crop_image_middle(landmarks, image):
    """Crop image

    Args:
        landmarks(numpy array, 68*2): Landmarks.
        image(PIL.Image): Input image.
    Return: Cropped image.
    Note:
        Yuval used this cropping method.

    """
    im_width, im_height = image.size

    landmarks = landmarks.astype('uint16')
    minx, miny = np.min(landmarks, 0)
    maxx, maxy = np.max(landmarks, 0)
    width, height = maxx - minx + 1, maxy - miny + 1
    centerx, centery = (minx + maxx) / 2, (miny + maxy) / 2
    avgx = int(round(np.sum(landmarks[:,0]) * 1.0 / landmarks.shape[0]))
    avgy = int(round(np.sum(landmarks[:,1]) * 1.0 / landmarks.shape[0]))
    devx, devy = centerx - avgx, centery - avgy
    dleft = int(round(0.1 * width)) + abs(min(devx, 0))
    dtop = int(round(height * (max(float(width) / height, 1.0) * 2 - 1))) \
           + abs(min(devy, 0))
    dright = int(round(0.1 * width)) + abs(max(devx, 0))
    dbottom = int(round(0.1 * height)) + abs(max(devy, 0))

    minx, miny = max(0, minx - dleft), max(0, miny - dtop)
    maxx = min(im_width - 1, maxx + dright)
    maxy = min(im_height - 1, maxy + dbottom)

    sq_width = max(maxx - minx + 1, maxy - miny + 1)
    centerx, centery = (minx + maxx) / 2, (miny + maxy) / 2
    minx = max(0, centerx - (sq_width - 1) / 2)
    miny = max(0, centery - (sq_width - 1) / 2)
    maxx = min(im_width - 1, minx + sq_width - 1)
    maxy = min(im_height - 1, miny + sq_width - 1)

    return image.crop((minx, miny, maxx, maxy)), landmarks - np.array((minx, miny))


def crop_image_min(landmarks, image):
    im_width, im_height = image.size
    landmarks = landmarks.astype('uint16')
    margin = 80
    margin = np.min((margin, np.min(landmarks), im_width-1-np.max(landmarks[:,0]), im_height-1-np.max(landmarks[:,1])))
    minx, miny = np.min(landmarks, 0) - margin
    minx, miny = max(minx, 0), max(miny, 0)
    maxx, maxy = np.max(landmarks, 0) + margin
    maxx, maxy = min(maxx, im_width-1), min(maxy, im_height-1)
    centerx, centery = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    if maxx - minx > maxy - miny:
        length = min(maxx - minx, im_height, im_width)
        topx = max(0, minx)
        if centery < length / 2.0:
            topy = 0
        elif im_width - centery < length / 2.0:
            topy = im_width - length
        else:
            topy = centery - length / 2.0
    else:
        length = min(maxy - miny, im_height, im_width)
        topy = max(0, miny)
        if centerx < length / 2.0:
            topx = 0
        elif im_height - centerx < length / 2.0:
            topx = im_height - length
        else:
            topx = centerx - length / 2.0
    minx, miny = int(topx), int(topy)
    landmarks[:,0] = np.clip(landmarks[:,0]-minx, 0, length-1)
    landmarks[:,1] = np.clip(landmarks[:,1]-miny, 0, length-1)
    return image.crop((minx, miny, minx+length, miny+length)), landmarks


def save_landmarks(filename, landmarks):
    """Save landmarks to file

    Note:
        Format:
        x1,y1
        x2,y2

    """
    try:
        file = open(filename, 'w')
    except OSError:
        print('cannot open', filename)

    for i in range(landmarks.shape[0]):
        file.write('%f,%f\n' % (landmarks[i,0], landmarks[i,1]))


def load_landmarks(filename, number=68):
    """Load landmarks from file

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


def show_result(image, mask, seg, save=False, filename='fig.png'):
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.imshow(image)
    ax2.imshow(mask)
    ax3.imshow(seg)

    if save:
        mask = Image.fromarray(np.uint8(mask*255))
        mask = mask.resize((224, 224))
        plt.imsave(filename, mask)
        #plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)


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
    d.addPairwiseBilateral(sxy=(30,30), srgb=(20,20,20), rgbim=im, compat=10)
    Q = d.inference(5)
    map = np.argmax(Q, axis=0).reshape((height,width))
    return map
