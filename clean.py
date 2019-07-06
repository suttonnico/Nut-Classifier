import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import imutils

def zero_pad(x,n):
    for i in range(1,5):
        if x < 10 ** i:
            return (n-i)*'0'+str(x)

def getNutId(x):
    return(x[4])

def subNutId(x,id):
    s = list(x)
    s[4] = id
    return "".join(s)

def getNutNumber(x):
    return(x[6:len(x)-4])

greenLower = (0, 0, 0)
greenUpper = (255, 150, 150)

def lp(img):
    size = 21
    smallBlur = np.ones((size, size), dtype="float") * (1.0 / (size * size))
    convoleOutput = cv2.filter2D(img,-1, smallBlur)
    return convoleOutput

def getNut(img):
    dif = 150
    empty = cv2.imread('data/empty/empty' + id + '.png')
    plt.figure()
    plt.imshow(img)
    plt.show()

    th = 140

    empty_gray = cv2.cvtColor(empty, cv2.COLOR_BGR2GRAY)
    trash, empty_BW = cv2.threshold(empty_gray, th, 255, cv2.THRESH_BINARY)

    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    plt.figure()
    plt.imshow(mask)
    plt.show()
    mask = cv2.erode(mask, None, iterations=10)
    mask = cv2.dilate(mask, None, iterations=10)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    print(cnts)

    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(img, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(img, center, 5, (0, 0, 255), -1)

    plt.figure()
    plt.imshow(img)
    plt.show()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = lp(gray)

    bausra, BW = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY)

if __name__ == '__main__':
    nut_dir_sep = 'selection/bad_01'
    nut_dir_sep_clean = 'selection/bad_01_out'
    good_or_bad = 0 #1= mala 0 = nuez buena

    pairs = {
        '0':'6',
        '2':'4'
    }

    new_imgs_files = [f for f in os.listdir(nut_dir_sep)]
    for f in new_imgs_files:
        id = getNutId(f)
        num = getNutNumber(f)
        print("ID: "+id)
        print("NUM: "+num)
        if id in pairs:
            img_org = cv2.imread(os.path.join(nut_dir_sep, f))
            th = 140

            img_gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
            bausra, imgBW = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)

            cv2.imwrite(os.path.join(nut_dir_sep_clean, f), imgBW)

