#!/usr/bin/env python

"""
Code for image morphology operations.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

def morphology(gray):
    kernels = [("rect 5x5", cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))),
                ("ellipse 5x5", cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))),
                ("ellipse 3x11", cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,11))),
                ("ellipse 11x5", cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,5))),
                ("cross 11x5", cv2.getStructuringElement(cv2.MORPH_CROSS,(11,5)))]

    ret_val, gray = cv2.threshold(gray, int(np.mean(gray)+np.std(gray)), 255, cv2.THRESH_BINARY)
    
    for kname, kernel in kernels:
        plt.figure()
        add_image(321, gray, 'Input')

        eroded = cv2.erode(gray, kernel=kernel)
        add_image(323, eroded, 'Eroded ' + kname)

        dilated = cv2.dilate(gray, kernel=kernel)
        add_image(324, dilated, 'Dilated ' + kname)

        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        add_image(325, opened, 'Opened ' + kname)

        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        add_image(326, closed, 'Closed ' + kname)

        plt.show(block=False)


def add_image(pos, img, title):
    plt.subplot(pos)
    plt.imshow(matplotlib_image(img))
    plt.title(title)


def custom_kernels(gray):
    """Morphological operations with custom kernels."""

    fat_backstroke = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0],
                               [0, 1, 1, 1, 0, 0, 0, 0, 0],
                               [0, 0, 1, 1, 1, 0, 0, 0, 0],
                               [0, 0, 0, 1, 1, 1, 0, 0, 0],
                               [0, 0, 0, 0, 1, 1, 1, 0, 0],
                               [0, 0, 0, 0, 0, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0, 0, 1, 1, 1]], np.uint8)

    fat_intersection = np.array([[1, 1, 1, 0, 0, 0, 1, 1, 1],
                                 [0, 1, 1, 1, 0, 1, 1, 1, 0],
                                 [0, 0, 1, 1, 1, 1, 1, 0, 0],
                                 [0, 0, 0, 1, 1, 1, 0, 0, 0],
                                 [0, 0, 1, 1, 1, 1, 1, 0, 0],
                                 [0, 1, 1, 1, 0, 1, 1, 1, 0],
                                 [1, 1, 1, 0, 0, 0, 1, 1, 1]], np.uint8)
    
    plt.figure()
    add_image(421, gray, 'Input')

    add_image(423, 255*fat_backstroke, 'Fat Backstroke kernel')
    add_image(424, 255*fat_intersection, 'Fat Intersection kernel')

    eroded = cv2.erode(gray, kernel=fat_backstroke)
    add_image(425, eroded, 'Backstroke eroded')
    eroded = cv2.erode(gray, kernel=fat_intersection)
    add_image(426, eroded, 'Intersection eroded')

    dilated = cv2.dilate(gray, kernel=fat_backstroke)
    add_image(427, dilated, 'Backstroke dilated')
    dilated = cv2.dilate(gray, kernel=fat_intersection)
    add_image(428, dilated, 'Intersection dilated')

    plt.show()


def matplotlib_image(image):
    """Convert an image to the colour RGB format used by matplotlib."""
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb



if  __name__ == '__main__':
    raw = cv2.imread('../images/digits.png')
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    plt.close('all')
    morphology(gray)

    symbols = cv2.imread('../images/symbols.png')
    gray = cv2.cvtColor(symbols, cv2.COLOR_BGR2GRAY)
    custom_kernels(gray)
