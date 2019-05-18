import imutils
import cv2
import numpy as np
import matplotlib.pyplot as plt


def zero_pad(x,n):
    for i in range(1,5):
        if x < 10 ** i:
            return (n-i)*'0'+str(x)

empty = cv2.imread('recinto1_L/nuez_0000.png')

#plt.figure()
#plt.imshow(empty[:,:,::-1])
#plt.show()
N = 80
empty_gray = cv2.cvtColor(empty, cv2.COLOR_BGR2GRAY)
th = 140
trash,empty_BW = cv2.threshold(empty_gray, th, 255, cv2.THRESH_BINARY)
for i in range(1,N):
    img = cv2.imread('recinto1_L/nuez_'+zero_pad(i,4)+'.png')

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    trash, img_BW = cv2.threshold(img_gray, th, 255, cv2.THRESH_BINARY)
    diff = empty_BW - img_BW
    if diff.sum() < 8000000:
        print('recinto1_L/nuez_' + zero_pad(i, 4) + '.png')
    print(i,diff.sum())



empty = cv2.imread('recinto1_R/nuez_0000.png')

#plt.figure()
#plt.imshow(empty[:,:,::-1])
#plt.show()
N = 80
empty_gray = cv2.cvtColor(empty, cv2.COLOR_BGR2GRAY)
th = 140
trash,empty_BW = cv2.threshold(empty_gray, th, 255, cv2.THRESH_BINARY)
for i in range(1,N):
    img = cv2.imread('recinto1_R/nuez_'+zero_pad(i,4)+'.png')

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    trash, img_BW = cv2.threshold(img_gray, th, 255, cv2.THRESH_BINARY)
    diff = empty_BW - img_BW
    if diff.sum() < 8000000:
        print('recinto1_R/nuez_' + zero_pad(i, 4) + '.png')
    print(i,diff.sum())