import numpy as np
import cv2
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
from sklearn.linear_model import LinearRegression
import math
import time
import matplotlib.pyplot as plt

rotation = True

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

def findRadius(img, empty):
    [N, M, D] = np.shape(img)
    div = 7
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
    #diff = diff+cv2.cvtColor(diff_pre, cv2.COLOR_BGR2GRAY) * 0.0001
    th = 30
    RG = RG > th
    RB = RB > th
    GB = GB > th
    mask = np.logical_or(RG, RB)
    mask = np.logical_or(mask, GB)
    diff = np.multiply(diff, mask)
    diff = np.clip(diff, 0, 255) / 255  # + diff_pre/255
    # diff =vdiff(diff_pre[:,:,0],diff_pre[:,:,1],diff_pre[:,:,2])
    # plt.figure()
    # plt.imshow(diff,cmap='gray')
    # plt.show()
    M = 50
    per = 0.2
    diff = cv2.resize(diff, (160, 120))
    vert = np.sum(diff, axis=1)
    vert = np.convolve(vert, np.blackman(M))
    th = np.max(vert)*per
    size = len(vert)-np.argmax(np.flip(vert,axis=0) > th)-np.argmax(vert>th)
    size_v = size/len(vert)*120*1.1
    hor = np.sum(diff, axis=0)
    hor = np.convolve(hor, np.blackman(M))
    th = np.max(vert) * per
    size = len(hor) - np.argmax(np.flip(hor, axis=0) > th) - np.argmax(hor > th)
    size_h = size / len(hor) * 160*1.1
    #print("sizev:"+str(size_v)+"    sizeH:"+str(size_h))
    #plt.figure()
    #plt.subplot(121)
    #plt.
    if size_h<size_v:
        return size_h
    else:
        return size_v

