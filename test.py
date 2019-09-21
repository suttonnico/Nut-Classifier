import sets_generator
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import imutils
from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from sklearn.linear_model import LinearRegression
import math
kernel = np.ones((3,3), np.uint8)
from keras.models import load_model
import cnn

def addUp(x,dif):
    #print('MAT')
    #print(x)
    a = np.average(x[0:dif,0:dif,0])
    b = np.average(x[0:dif,0:dif,1])
    c = np.average(x[0:dif, 0:dif,2])
    #print(a,b,c)
    return [a,b,c]
def diffInColor(x,y):
    x= [int(x[0]),int(x[1]),int(x[2])]
    y = [int(y[0]), int(y[1]), int(y[2])]
   # print('X:'+str(x))
   # print('Y:' + str(y))

    R = np.abs(x[0] - y[0])
    G = np.abs(x[1] - y[1])
    B = np.abs(x[2] - y[2])
  #  print([R,G,B])
    dif = (R + G + B)/3
    th = 40

    if (np.abs(R-G)<th) & (np.abs(R-B) <th) & (np.abs(G-B) <th):
        dif = dif*0.001
    if dif > 255:
        dif = 255
    dif = dif/255
    return dif
"""
def findRadius(img,empty):
    #plt.figure()
    #plt.imshow(img)
    #plt.show()
    kernel = np.ones((5, 5), np.uint8)
    [N, M, D] = np.shape(img)
    diff = np.zeros([N, M])
    step = 1
    start = time.time()
    for i in range(int(N / step)):
        for j in range(int(M / step)):
            diff[i * step:i * step + step, j * step:j * step + step] = diffInColor(
                addUp(img[i * step:i * step + step, j * step:j * step + step], step),
                addUp(empty[i * step:i * step + step, j * step:j * step + step], step))
    its = 3
    diff = cv2.dilate(diff, kernel, iterations=its)
    diff = cv2.erode(diff, None, iterations=its)
    #plt.figure()
    #plt.imshow(diff, cmap='gray')
    #plt.show()
    th = 0.2
    #print(diff)
    trash, diff = cv2.threshold(diff, th, 1, cv2.THRESH_BINARY)
    #plt.figure()
    #plt.imshow(diff, cmap='gray')
    #plt.show()
    edges = canny(diff, sigma=6)
    [y_edges,x_edges] = np.nonzero(edges)
    y_min = np.min(y_edges)
    y_max = np.max(y_edges)
    x_min = np.min(x_edges)
    x_max = np.max(x_edges)
#    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
    plt.figure()
    plt.subplot(121)
    plt.imshow(edges,cmap='gray')
    plt.subplot(122)
    plt.imshow(img)
    plt.show()


    hough_radii = np.arange(10, 30, 5)
    hough_res = hough_circle(edges, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=10)
    maxR = np.argmax(radii)
    print("Radio Máximo:" + str(np.argmax(radii)))
    print("BBBBB")

    # Draw them
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    image = color.gray2rgb(img)

    #circy, circx = circle_perimeter(cy[maxR], cx[maxR], radii[maxR])
    #image[circy, circx] = (220, 20, 20)

#    ax.imshow(image, cmap=plt.cm.gray)
 #   plt.show()
    dif = 50
    yMin = 0
    yMax = N
    xMin = 0
    xMax = M
    if cy[maxR]-dif > 0:
        yMin = cy[maxR]-dif
    if cy[maxR]+dif < N:
        yMax = cy[maxR] + dif
    if cx[maxR] - dif > 0:
        xMin = cx[maxR] - dif
    if cx[maxR] + dif < M:
        xMax = cx[maxR] + dif
    nutImg = diff[yMin:yMax,xMin:xMax]

    #plt.figure()
    #plt.imshow(nutImg,cmap='gray')
    #plt.show()
    [y_n,x_n] = np.nonzero(nutImg)
    y_n = y_n.reshape(-1,1)
    x_n = x_n.reshape(-1, 1)
    print("Y: "+str(np.shape(y_n)))
    print("X: " + str(np.shape(x_n)))
    lm3 = LinearRegression()
    lm3.fit(x_n, y_n)
    x = np.arange(0,2*dif, 0.1)
    y =lm3.coef_[0] * x
    y = y+lm3.intercept_
    plt.figure()
    plt.scatter(x_n,y_n)
    plt.plot(x,y,color='r')
    plt.show()
    rows=N
    cols = M

    Mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), math.atan(lm3.coef_[0])*360/2/3.1415, 1)
    rot_img = cv2.warpAffine(img, Mat, (M, N))

    rot_empty = cv2.warpAffine(empty, Mat, (M, N))

    findRadius(rot_img, rot_empty)

    return radii[maxR]
"""
import size_classification
import cv2

model = load_model('model_backup.h5')
weights = model.get_weights()

size = 150
W = 2*size
H = 2*size

my_cnn = cnn.cnn(img_width=W, img_height=H)
my_cnn.set_weights(weights)

img1 = cv2.imread('data_cinta/good__02/nuez0_000050.png')
img2 = cv2.imread('test/data/nuez6_000000.png')
img = np.concatenate((img1, img2), axis=1)
img = cv2.resize(img, (4 * size, 2 * size))

pred = my_cnn.predict_classes(img.reshape([-1, 300, 600, 3]), batch_size=1)
print(pred)
exit(1111)
empty1 = cv2.imread('data_cinta/empty/empty0.png')
empty2 = cv2.imread('test/data/empty6.png')
print("start")
size1 = findRadius(img1, empty1)
print("start")
size2 = findRadius(img2, empty2)
print("pixeles camara 1:" + str(size1))
print("pixeles camara 2:" + str(size2))
print("Diametro: " + str(size_classification.sizes2rad(size1, size2, 120)))


