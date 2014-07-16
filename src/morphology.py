#!/usr/bin/env python

'''
Code for image morphology operations.
'''

import cv2
import matplotlib
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
        ax_image = plt.subplot(321)
        plt.imshow(matplotlib_image(gray))
        plt.title('Input')
    
        eroded = cv2.erode(gray, kernel=kernel)
        plt.subplot(323)
        plt.imshow(matplotlib_image(eroded))
        plt.title('Eroded ' + kname)
        
        dilated = cv2.dilate(gray, kernel=kernel)
        plt.subplot(324)
        plt.imshow(matplotlib_image(dilated))
        plt.title('Dilated '+ kname)

        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        plt.subplot(325)
        plt.imshow(matplotlib_image(opened))
        plt.title('Opened '+ kname)

        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        plt.subplot(326)
        plt.imshow(matplotlib_image(closed))
        plt.title('Closed ' + kname)
    
        plt.show(block=False)


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
