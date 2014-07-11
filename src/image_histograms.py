#!/usr/bin/env python

'''
Code for drawing image intensity histograms along the rows and columns.
'''

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def histograms(gray):
    h,w = gray.shape[0:2]

    all_stddev = np.std(gray)
    all_mean = np.mean(gray)

    ROW_AXIS = 1
    COL_AXIS = 0
    
    row_stddev = np.std(gray, ROW_AXIS)
    row_mean = np.mean(gray, ROW_AXIS)

    col_stddev = np.std(gray, COL_AXIS)
    col_mean = np.mean(gray, COL_AXIS)

    plt.figure()

    ax_image = plt.subplot(223)
    ax_image.set_xticks(range(0,w+1,100))
    ax_image.set_yticks(range(0,h+1,50))
    plt.imshow(matplotlib_image(gray))

    ax_cols = plt.subplot(221, sharex=ax_image)
    plt.plot(col_stddev, 'r', label='std')
    plt.plot(col_mean, 'b', label='mean')
    plt.hlines([np.mean(col_stddev)], 0,w-1, 'r')
    plt.hlines([np.mean(col_mean)], 0,w-1, 'b')
    ax_cols.legend()
    plt.title('By column')

    ax_rows = plt.subplot(224, sharey=ax_image)
    # rotate it by changing xs and ys
    xs = range(h)
    plt.plot(row_stddev,xs, 'r', label='std')
    plt.plot(row_mean, xs,'b', label='mean')
    plt.vlines([np.mean(row_stddev)], 0,h-1, 'r')
    plt.vlines([np.mean(row_mean)], 0,h-1, 'b')
    ax_rows.legend()
    plt.title('By row')

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
    histograms(gray)
