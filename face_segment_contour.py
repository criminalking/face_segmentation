#!/usr/bin/env python
import argparse
from scipy.spatial import ConvexHull
from skimage import draw
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from face_alignment.api import FaceAlignment, LandmarksType, NetworkSize
from util import read_list, show_result, CRF

def main(args):
    image_paths = read_list(args.image_list)
    for path in image_paths:
        im = Image.open(path)
        width, height = im.size

        # use 2D-FAN detect landmarks
        fa = FaceAlignment(LandmarksType._2D, enable_cuda=True,
                           flip_input=False, use_cnn_face_detector=True)
        landmarks = fa.get_landmarks(np.array(im))[-1]
        landmarks[:,1] = height - landmarks[:,1]

        # generate a contour curve with contour points
        hull = ConvexHull(landmarks)
        # draw landmarks
        lm = np.array(im)
        for i in range(landmarks.shape[0]):
            rr, cc = draw.circle(height-landmarks[i,1].astype('int32'), landmarks[i,0].astype('int32'), 5)
            lm[rr, cc, :] = np.array((255, 0, 0))
        # create mask
        mask = np.zeros((height, width))
        rr, cc = draw.polygon(height-landmarks[hull.vertices,1], landmarks[hull.vertices,0], mask.shape)
        mask[rr,cc] = 1

        save = True if args.save == 'True' else False
        path = path[:-1] if path[:-1] == '/' else path
        image_name = path[path.rindex('/')+1:-4] + '_contour_nocrf.png'
        show_result(lm, mask, np.tile((mask!=0)[:,:,np.newaxis], (1,1,3)) * im,
                    save=save, filename='images/'+image_name)

        # add CRF
        prob = np.concatenate(((1-mask)[np.newaxis,:,:]*0.9 +
                               mask[np.newaxis, :, :]*0.1,
                               mask[np.newaxis, :, :]*0.9 +
                               (1-mask)[np.newaxis, :, :]*0.1), axis=0)
        map = CRF(prob, np.array(im))
        image_name = path[path.rindex('/')+1:-4] + '_contour_crf.png'
        show_result(im, map, np.tile((map!=0)[:,:,np.newaxis], (1,1,3)) * im,
                    save=save, filename='images/'+image_name)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=
                                     'Face segmentation with landmarks.')
    parser.add_argument('--image_list',
                        default='input/list.txt',
                        type=str, help='path to image file')
    parser.add_argument('--save', choices=['True', 'False'],
                        default='False', help='choose if save final result')
    args = parser.parse_args()

    main(args)





