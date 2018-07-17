"""This file is used to generate contour with landmarks
"""

import argparse
from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_landmarks(filename):
    landmarks = np.zeros((68, 2))
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


def generate_contour(points):
    """Find convex hull of points

    Args:
        points (numpy array, N*2): Landmarks points.
    """
    #points = points / np.max(points, 0) - 0.01
    hull = ConvexHull(points)

    plt.plot(points[:,0], points[:,1], 'o')
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
    # We could also have directly used the vertices of the hull,
    # which for 2-D are guaranteed to be in counterclockwise order:
    plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
    plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
    plt.axis('scaled')
    plt.show()
    #plt.savefig('contour.png', bbox_inches='tight')


def main(args):
    landmarks_file = args.landmarks
    image_file = args.image
    im = Image.open(image_file)
    height, width = im.size
    landmarks = load_landmarks(landmarks_file)
    landmarks[:,1] = height - landmarks[:,1]
    # select contour points
    #contour_points = get_contour_side(landmarks)
    # generate a contour curve with contour points
    generate_contour(landmarks)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=
                                 'Generate contour of landmarks.')
    parser.add_argument('--landmarks',
        default='input/landmarks.txt',
        type=str, help='path to landmarks file')
    parser.add_argument('--image',
        default='input/face.png',
        type=str, help='path to landmarks file')
    args = parser.parse_args()

    main(args)





