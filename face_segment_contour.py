#!/usr/bin/env python
import argparse
from scipy.spatial import ConvexHull
from skimage import draw
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from util import read_list, load_landmarks, show_result, CRF

def main(args):
    image_paths = read_list(args.image_list)
    for image_file in image_paths:
        # landmarks_file should have the same prefix as image_file
        landmarks_file = image_file[:-3] + 'txt'
        im = Image.open(image_file)
        width, height = im.size
        landmarks = load_landmarks(landmarks_file)
        landmarks[:,1] = height - landmarks[:,1]
        # select contour points
        #contour_points = get_contour_side(landmarks)
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
        show_result(lm, mask, np.tile((mask!=0)[:,:,np.newaxis], (1,1,3)) * im)

        # add CRF
        prob = np.concatenate(((1-mask)[np.newaxis,:,:]*0.9 +
                               mask[np.newaxis, :, :]*0.1,
                               mask[np.newaxis, :, :]*0.9 +
                               (1-mask)[np.newaxis, :, :]*0.1), axis=0)
        map = CRF(prob, np.array(im))
        show_result(im, map, np.tile((map!=0)[:,:,np.newaxis], (1,1,3)) * im)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=
                                     'Face segmentation with landmarks.')
    parser.add_argument('--image_list',
                        default='input/list.txt',
                        type=str, help='path to image file')
    args = parser.parse_args()

    main(args)





