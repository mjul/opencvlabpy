#!/usr/bin/env python

"""
Code for Harris corner detection.
"""

import cv2
import numpy as np


def interactive_harris(title, img):
    cv2.imshow(title, img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    def update_harris(pos):
        bs_i = cv2.getTrackbarPos('bs', title)
        ks_i = cv2.getTrackbarPos('ks', title)
        k_i = cv2.getTrackbarPos('k', title)
        odds = [2*x+1 for x in range(100)]
        bs = odds[bs_i]
        ks = odds[ks_i]
        k = k_i
        harris = cv2.cornerHarris(gray, blockSize=bs, ksize=ks, k=k)
        harris = cv2.normalize(harris, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
        print "%s :: bs=%d, ks=%d, k=%d" % (title, bs, ks, k)
        cv2.imshow(title, np.vstack((harris,gray)))
    cv2.createTrackbar('bs', title, 0, 20, update_harris)
    cv2.createTrackbar('ks', title, 0, 15, update_harris)
    cv2.createTrackbar('k', title, 0, 100, update_harris)
    update_harris(None)


if  __name__ == '__main__':
    digits = cv2.imread('../images/digits.png')
    interactive_harris('digits', digits)

    symbols = cv2.imread('../images/symbols.png')
    interactive_harris('symbols', symbols)

    print "Done. Press enter."
    cv2.waitKey()
