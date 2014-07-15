#!/usr/bin/env python

'''
Code for drawing image intensity histograms along the rows and columns.
'''

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

def find_lines(gray):
    h,w = gray.shape[0:2]

    eq = cv2.equalizeHist(gray)
    canny = cv2.Canny(eq, 100, 200, apertureSize=3)
    
    # rho: distance resolution in pixels
    # theta: angle resolution in radians
    # threshold: minimum number of votes for line
    lines = cv2.HoughLinesP(canny, rho=1, theta=math.pi/180, threshold=100, minLineLength=200, maxLineGap=50)
    print canny.shape, lines.shape
    
    with_lines = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for i in range(lines.shape[1]):
        line = lines[0][i]
        cv2.line(with_lines, tuple(line[0:2]), tuple(line[2:4]), (0,0,255), 3, 8)
    
    plt.figure()

    ax_image = plt.subplot(221)
    plt.imshow(matplotlib_image(gray))
    plt.title('Input image')

    plt.subplot(222)
    plt.imshow(matplotlib_image(eq))
    plt.title('Pre-edge')

    plt.subplot(223)
    plt.imshow(matplotlib_image(canny))
    plt.title('Edges')

    plt.subplot(224)
    plt.imshow(matplotlib_image(with_lines))
    plt.title('Hough lines')

    plt.show(block=False)


def matplotlib_image(image):
    """Convert an image to the colour RGB format used by matplotlib."""
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb

def scale_down(img):
    blurred = cv2.GaussianBlur(img, (3,3), 3)
    small = cv2.resize(blurred, (480,640))
    return small

if  __name__ == '__main__':
    raw = cv2.imread('../images/electricity_meter.jpg')
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    small = scale_down(gray)
    plt.close('all')
    find_lines(small)
