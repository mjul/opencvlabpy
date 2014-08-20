#!/usr/bin/env python

'''
Calculate local min and max using dilation and erosion operations.
'''

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random

def extrema(image, kernel_size=(11,11)):
    """Find local extrema in the image."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    
    # Dilation is a local max operation over the kernel area, erosion is local min
    # Highlight the points that match the max value in the neighbourhood
    local_max = cv2.compare(image, cv2.dilate(image, kernel=kernel, iterations=1), cv2.CMP_EQ)
    local_min = cv2.compare(image, cv2.erode(image, kernel=kernel, iterations=1), cv2.CMP_EQ)
    non_min_maxes = cv2.threshold(cv2.compare(local_max, local_min, cv2.CMP_GT), 1, 255, cv2.THRESH_BINARY)[1]
    
    fig = plt.figure()
    show(411, image, 'Input')
    show(412, local_max, 'Pixel equals Local max')
    show(413, local_min, 'Pixel equals Local min')
    show(414, non_min_maxes, 'Non-min maxes')
    plt.axis('off')
    plt.show(block=False)


def show(pos, img, title):
    plt.subplot(pos)
    plt.imshow(matplotlib_image(img))
    plt.title(title)
    plt.axis('off')
    

def matplotlib_image(image):
    """Convert an image to the colour RGB format used by matplotlib."""
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb


if  __name__ == '__main__':
    w,h = 200, 50
    gray = np.zeros((h,w), np.uint8)
    for i in range(10):
        x,y = random.randrange(w), random.randrange(h)
        r = random.randrange(10, 20)
        col = random.randrange(255)
        cv2.circle(gray, (x,y), r, col, -1)
        
    extrema(gray, (11,11))
