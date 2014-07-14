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
    
    ax_image = plt.subplot(323)
    ax_image.set_xticks(range(0,w+1,100))
    ax_image.set_yticks(range(0,h+1,50))
    plt.imshow(matplotlib_image(gray))

    ax_cols = plt.subplot(321, sharex=ax_image)
    plt.plot(col_stddev, 'r', label='std')
    plt.plot(col_mean, 'b', label='mean')
    plt.hlines([np.mean(col_stddev)], 0,w-1, 'r')
    plt.hlines([np.mean(col_mean)], 0,w-1, 'b')
    ax_cols.legend()
    plt.title('By column')

    ax_rows = plt.subplot(324, sharey=ax_image)
    # rotate it by changing xs and ys
    xs = range(h)
    plt.plot(row_stddev,xs, 'r', label='std')
    plt.plot(row_mean, xs,'b', label='mean')
    plt.vlines([np.mean(row_stddev)], 0,h-1, 'r')
    plt.vlines([np.mean(row_mean)], 0,h-1, 'b')
    ax_rows.legend()
    plt.title('By row')

    # High standard deviation blocks
    cs = moving_mean(col_stddev, 5)
    rs = moving_mean(row_stddev, 5)
    col_mean = np.mean(col_stddev) * 1.0
    row_mean = np.mean(row_stddev) * 0.7
    high_std_cols = [x for x in range(w) if cs[x] >= col_mean]
    high_std_rows = [y for y in range(h) if rs[y] >= row_mean]
    shrunk = gray[:,high_std_cols]
    shrunk = shrunk[high_std_rows,:]
    plt.subplot(325)
    plt.imshow(matplotlib_image(shrunk))
    plt.title('High standard deviation blocks')

    smart = shrink_low_standard_deviation_stripes(gray)
    plt.subplot(326)
    plt.imshow(matplotlib_image(smart))
    plt.title('Reduced low standard-deviation stripes')
    
    plt.show()


def matplotlib_image(image):
    """Convert an image to the colour RGB format used by matplotlib."""
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb


def moving_mean(xs, window_size=10):
    """Calculate the mean of a window around each point x in the range xs."""
    return [np.mean(xs[(x-window_size/2):(x+window_size/2)]) for x in range(len(xs))]
    

def shrink_low_standard_deviation_stripes(img):
    # Use the standard deviations for the original image
    # for shrinking both directions
    column_stds = np.std(img,0)
    row_stds = np.std(img,1)
    reduced_x = shrink_low_standard_deviation_columns(img, column_stds, 1.2, 3.0)
    reduced_xy = shrink_low_standard_deviation_rows(reduced_x, row_stds, 0.8, 3.0)
    return reduced_xy

def shrink_low_standard_deviation_columns(img, column_stds=None, mean_threshold_factor=1.0, cumulative_threshold_factor=5.0):
    select_column_range = lambda i1,i2: img[:,i1:i2]
    h = img.shape[0]
    if column_stds is None:
        column_stds = np.std(img, 0)
    slices = slice_image(column_stds, 1, select_column_range, mean_threshold_factor, cumulative_threshold_factor)
    return np.hstack([slice.astype(np.uint8).reshape((h,1)) for slice in slices])

def shrink_low_standard_deviation_rows(img, row_stds=None, mean_threshold_factor=1.0, cumulative_threshold_factor=5.0):
    select_row_range = lambda i1,i2: img[i1:i2,:]
    if row_stds is None:
        row_stds = np.std(img, 1)
    slices = slice_image(row_stds, 0, select_row_range, mean_threshold_factor, cumulative_threshold_factor)
    w = img.shape[1]
    return np.vstack([slice.astype(np.uint8).reshape((1,w)) for slice in slices])


def slice_image(stds, avg_axis, select_slice, mean_threshold_factor, cumulative_threshold_factor):
    """
    Cut image into slices according to the standard deviations.
    If the standard deviation for a slice is above the mean 
    the slice is transfered verbatim, otherwise it is averaged with
    its adjacent low-standard-deviation slices.
    """
    mean_std = np.mean(stds)
    end_of_image = stds.shape[0]
    slices = []
    cursor = 0
    max_slice_cumsum = cumulative_threshold_factor*mean_std
    verbatim_threshold = mean_std * mean_threshold_factor
    take_verbatim = lambda i: stds[i] >= verbatim_threshold
    continue_slice = False
    while (cursor < end_of_image):
        slice = None
        if take_verbatim(cursor):
            # Take high standard deviation slices
            slice = select_slice(cursor,cursor+1)
            cursor += 1
        else:
            # Average out low std.dev. slices
            cumsum = 0
            begin = cursor
            while (cursor < end_of_image) and (not take_verbatim(cursor)) and (cumsum < max_slice_cumsum):
                cumsum += stds[cursor]
                cursor += 1
            end = cursor
            avg_slice = np.mean(select_slice(begin,end), avg_axis)
            slice = avg_slice
        slices.append(slice)
    return slices


if  __name__ == '__main__':
    raw = cv2.imread('../images/digits.png')
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    plt.close('all')
    histograms(gray)
