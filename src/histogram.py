#!/usr/bin/env python

'''
Code for drawing image colour histogram.
'''

# OpenCV imports
import numpy as np
import cv2

# Histogram and plotting
import matplotlib.pyplot as plt
import matplotlib as mpl


# ----------------------------------------------------------------
# Histograms
# Taken from http://opencv-code.com/tutorials/drawing-histogram-in-python-with-matplotlib/
# ----------------------------------------------------------------

def show_histogram(img):
    """ Function to display image histogram. 
        Supports single and three channel images. """

    if img.ndim == 2:
        # Input image is single channel
        plt.hist(img.flatten(), 256, range=(0, 250), fc='k')
        plt.show()

    elif img.ndim == 3:
        # Input image is three channels
        fig = plt.figure()
        fig.add_subplot(311)
        plt.hist(img[...,0].flatten(), 256, range=(0, 250), fc='b')
        fig.add_subplot(312)
        plt.hist(img[...,1].flatten(), 256, range=(0, 250), fc='g')
        fig.add_subplot(313)
        plt.hist(img[...,2].flatten(), 256, range=(0, 250), fc='r')
        plt.show()


# ----------------------------------------------------------------

def show_hsv_histogram(img):
    '''Show a histogram of the Hue values of the HSV representation of the image.'''
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v, = cv2.split(hsv)

    n, bins, patches = plt.hist(h.flatten(), 180, normed=True)
    
    # Colour histogram bins according to the H value
    cmap = mpl.cm.hsv
    b_max = float(max(bins))
    for b,patch in zip(bins, patches):
        # scale bins to 0-1.0 for colour map look-up
        c = cmap(b/b_max) 
        patch.set_color(c)

    plt.show()
    
# ----------------------------------------------------------------

if  __name__ == '__main__':
    img = cv2.imread('../images/paper_on_table.jpg')
    show_histogram(img)
    show_hsv_histogram(img)
    cv2.waitKey()
    cv2.destroyAllWindows()
