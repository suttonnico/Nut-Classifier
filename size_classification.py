import numpy as np
import cv2
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
from sklearn.linear_model import LinearRegression
import math
import time
import matplotlib.pyplot as plt

def sizes2rad(size1, size2, T):
    tv = 0.51
    L = 20
    # print(T)
    t1 = math.atan(size1 / T * math.tan(tv))
    # print("tita 1:" + str(t1))
    t2 = math.atan(size2 / T * math.tan(tv))
    # print("tita 2:" + str(t2))
    h1 = L / (math.cos(t1) + math.sin(t1) / math.tan(t2))
    return math.sin(t1) * h1


def addUp(x, dif):
    a = np.average(x[0:dif, 0:dif, 0])
    b = np.average(x[0:dif, 0:dif, 1])
    c = np.average(x[0:dif, 0:dif, 2])
    return [a, b, c]


def diffInColor(x, y):
    x = [int(x[0]), int(x[1]), int(x[2])]
    y = [int(y[0]), int(y[1]), int(y[2])]
    # print('X:'+str(x))
    # print('Y:' + str(y))

    R = np.abs(x[0] - y[0])
    G = np.abs(x[1] - y[1])
    B = np.abs(x[2] - y[2])
    #  print([R,G,B])
    dif = (R + G + B) / 3
    th = 40

    if (np.abs(R - G) < th) & (np.abs(R - B) < th) & (np.abs(G - B) < th):
        dif = dif * 0.01
    if dif > 255:
        dif = 255
    dif = dif / 255
    return dif


def diffInColor2(R, G, B):
    R = int(R)
    G = int(G)
    B = int(B)
    dif = (R + G + B) / 3
    th = 40

    if (np.abs(R - G) < th) & (np.abs(R - B) < th) & (np.abs(G - B) < th):
        dif = dif * 0.001
    if dif > 255:
        dif = 255
    dif = dif / 255
    return dif

def findEdges(img, empty):
    test_id = 191
    start = time.time()
    kernel = np.ones((5, 5), np.uint8)
    [N, M, D] = np.shape(img)
    diff = np.zeros([N, M])
    step = 3

    if True:
        div = 3
        img_small = cv2.resize(img, (int(160 / div), int(120 / div))).astype(int)
        empty_small = cv2.resize(empty, (int(160 / div), int(120 / div))).astype(int)
        diff_pre = np.absolute(img_small - empty_small)

        R = diff_pre[:, :, 0]
        G = diff_pre[:, :, 1]
        B = diff_pre[:, :, 2]
        RG = np.abs(R - G)
        RB = np.abs(R - B)
        GB = np.abs(G - B)
        diff = (R + G + B) / 3
        diff_pre = diff * 0.001
        th = 40
        RG = RG > th
        RB = RB > th
        GB = GB > th
        mask = np.logical_or(RG,RB)
        mask = np.logical_or(mask, GB)
        diff = np.multiply(diff, mask)
        diff = np.clip(diff, 0, 255)/255 + diff_pre/255
        # diff =vdiff(diff_pre[:,:,0],diff_pre[:,:,1],diff_pre[:,:,2])
        diff = cv2.resize(diff, (160, 120))
        #print(diff)
        """
        diff = np.zeros([N, M])
        for i in range(int(N / step)):
            for j in range(int(M / step)):
                diff[i * step:i * step + step, j * step:j * step + step] = diffInColor(
                    addUp(img[i * step:i * step + step, j * step:j * step + step], step),
                    addUp(empty[i * step:i * step + step, j * step:j * step + step], step))
        
        print('DIFF1')
        print(np.max(diff_1))
        print('diff_2')
        print(np.max(diff))
        print('DIV')
        print(np.divide(diff,diff_1))
        plt.figure()
        plt.subplot(121)
        plt.imshow(diff,cmap='gray')
        plt.subplot(122)
        plt.imshow(diff_1, cmap='gray')
        plt.show()
        """
    #print("tiempo en la clasifiacion por imagenes: " + str(time.time() - start))
    its = 3
    diff = cv2.dilate(diff, kernel, iterations=its)
    diff = cv2.erode(diff, None, iterations=its)

    th = 0.3
    edges = canny(diff, sigma=6)
    diff_pre = diff
    while not edges.any():
        trash, diff_after = cv2.threshold(diff_pre, th, 1, cv2.THRESH_BINARY)
        edges = canny(diff_after, sigma=6)
        th = th / 2
    return edges,diff

def findRadius(img, empty):
    [N, M, D] = np.shape(img)

    Mat = cv2.getRotationMatrix2D((M / 2, N / 2), findRot(img,empty), 1)
    rot_img = cv2.warpAffine(img, Mat, (M, N))

    rot_empty = cv2.warpAffine(empty, Mat, (M, N))

    [edges,diff] = findEdges(rot_img,rot_empty)
    [y_edges, x_edges] = np.nonzero(edges)
    y_min = np.min(y_edges)
    y_max = np.max(y_edges)

    return y_max - y_min


def findRot(img, empty):
    [N, M, D] = np.shape(img)
    [edges,diff] = findEdges(img, empty)

    diff_pre = diff
    while not edges.any():
        trash, diff_after = cv2.threshold(diff_pre, th, 1, cv2.THRESH_BINARY)
        edges = canny(diff_after, sigma=6)
        th = th / 2
    [y_edges, x_edges] = np.nonzero(edges)
    y_min = np.min(y_edges)
    y_max = np.max(y_edges)
    x_min = np.min(x_edges)
    x_max = np.max(x_edges)
    # cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)

    hough_radii = np.arange(10, 30, 5)
    hough_res = hough_circle(edges, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=10)
    maxR = np.argmax(radii)
    dif = 50
    yMin = 0
    yMax = N
    xMin = 0
    xMax = M
    if cy[maxR] - dif > 0:
        yMin = cy[maxR] - dif
    if cy[maxR] + dif < N:
        yMax = cy[maxR] + dif
    if cx[maxR] - dif > 0:
        xMin = cx[maxR] - dif
    if cx[maxR] + dif < M:
        xMax = cx[maxR] + dif
    nutImg = diff[yMin:yMax, xMin:xMax]

    # plt.figure()
    # plt.imshow(nutImg,cmap='gray')
    # plt.show()
    [y_n, x_n] = np.nonzero(nutImg)
    y_n = y_n.reshape(-1, 1)
    x_n = x_n.reshape(-1, 1)
    lm3 = LinearRegression()
    lm3.fit(x_n, y_n)


    return math.atan(lm3.coef_[0]) * 360 / 2 / 3.1415
