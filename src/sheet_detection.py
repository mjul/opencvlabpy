#!/usr/bin/env python

'''
Code for detecting a sheet of paper in an image.
See http://stackoverflow.com/questions/8667818/opencv-c-obj-c-detecting-a-sheet-of-paper-square-detection
'''

# OpenCV imports
import numpy as np
import cv2

# ----------------------------------------------------------------

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def approx_poly(cnt):
    '''Approximate a contour with a polygon.'''
    cnt_len = cv2.arcLength(cnt, True)
    return cv2.approxPolyDP(cnt, 0.02*cnt_len, True)

def max_cos(poly):
    '''Get the max cosine of the angles between three successive points in the quadrilateral, poly.'''
    points = poly.reshape(-1, 2)
    max_cos = np.max([angle_cos(points[i], points[(i+1) % 4], points[(i+2) % 4] ) for i in xrange(4)])

def draw_contours(img, contours):
    '''Draw the contours on the image, returning a new image.'''
    result = img.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 3)
    return result


def show(title, img):
    if img.shape == (640,480):
        cv2.imshow(img)
    else:
        resized = cv2.resize(cv2.GaussianBlur(img, (3,3), .5), (640,480))
        cv2.imshow(title, resized)

    
def find_page_with_morphology(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, 3)
    dilated = cv2.dilate(median, kernel=None, iterations=2)
    eroded = cv2.erode(median, kernel=None, iterations=1)
    gradient = dilated - eroded
    show('Gradient', gradient)
    # Compute the best threshold to separate two peaks of the histogram (OTSU method)
    optimal_threshold, otsu = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY or cv2.THRESH_OTSU)
    ret_val, binarized = cv2.threshold(gradient, optimal_threshold, 255, cv2.THRESH_BINARY)

    show('Binarized', binarized)
    
    contours, hierarchy = cv2.findContours(binarized, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    polys = [approx_poly(cnt) for cnt in contours]

    poly_img = draw_contours(img, polys)
    show('All polys', poly_img)
    
    min_area = 50*50
    cos_epsilon = .1
    quads = [poly for poly in polys
                if len(poly) == 4
                and cv2.contourArea(poly) > min_area
                and cv2.isContourConvex(poly)
                and max_cos(poly) < cos_epsilon]

    quads_img = draw_contours(img, quads)
    show('Larger rectangluar polys', quads_img)
    largest = sorted(quads, key=lambda q: cv2.contourArea(q))[-1]

    page_img = draw_contours(img, [largest])
    show('Page', page_img)


if  __name__ =='__main__':
    img = cv2.imread('../images/paper_on_table.jpg')
    small = cv2.resize(cv2.GaussianBlur(img, (3,3), .5), (640,480))
    find_page_with_morphology(small)
    cv2.waitKey()
    cv2.destroyAllWindows()
