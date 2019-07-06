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

kernel = np.ones((3,3), np.uint8)
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

def findRadius(img,empty):
    plt.figure()
    plt.subplot()
    plt.subplot(211)
    plt.imshow(img)
    plt.subplot(212)
    plt.imshow(empty)
    plt.show()
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
    plt.figure()
    plt.imshow(diff, cmap='gray')
    plt.show()
    th = 0.2
    print(diff)
    trash, diff = cv2.threshold(diff, th, 1, cv2.THRESH_BINARY)
    plt.figure()
    plt.imshow(diff, cmap='gray')
    plt.show()
    edges = canny(diff, sigma=6)

    plt.figure()
    plt.imshow(edges)
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

    circy, circx = circle_perimeter(cy[maxR], cx[maxR], radii[maxR])
    image[circy, circx] = (220, 20, 20)

    ax.imshow(image, cmap=plt.cm.gray)
    plt.show()
    dif = 40
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

    plt.figure()
    plt.imshow(nutImg,cmap='gray')
    plt.show()
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
    return radii[maxR]

"""
sin = cv2.imread('data/recinto/sin3.jpg')
con = cv2.imread('data/recinto/con3.jpg')

[N,M,D] = np.shape(sin)
diff = np.zeros([N,M])
step = 10
start = time.time()
for i in range(int(N/step)):
    for j in range(int(M/step)):
        diff[i*step:i*step+step,j*step:j*step+step] = diffInColor(addUp(con[i*step:i*step+step,j*step:j*step+step],step),addUp(sin[i*step:i*step+step,j*step:j*step+step],step))
its = 15
diff= cv2.dilate(diff, kernel,iterations=its)
diff = cv2.erode(diff, None, iterations=its)
plt.figure()
plt.imshow(diff,cmap='gray')
plt.show()
th = 0.4
print(diff)
trash,diff = cv2.threshold(diff, th, 1, cv2.THRESH_BINARY)
plt.figure()
plt.imshow(diff,cmap='gray')
plt.show()
edges = canny(diff, sigma=6)


plt.figure()
plt.imshow(edges)
plt.show()

hough_radii = np.arange(100, 720, 50)
hough_res = hough_circle(edges, hough_radii)
accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                           total_num_peaks=10)
maxR= np.argmax(radii)
print("Radio Máximo:"+str(np.argmax(radii)))
print("BBBBB")


# Draw them
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
image = color.gray2rgb(con)

circy, circx = circle_perimeter(cy[maxR], cx[maxR], radii[maxR])
image[circy, circx] = (220, 20, 20)

ax.imshow(image, cmap=plt.cm.gray)
plt.show()
exit(1111)
result = hough_ellipse(edges, accuracy=1000, threshold=20,
                       min_size=10, max_size=1200)
print("AAAAAAAAAA")
result.sort(order='accumulator')

best = list(result[-1])
yc, xc, a, b = [int(round(x)) for x in best[1:5]]
orientation = best[5]

# Draw the ellipse on the original image
cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
con[cy, cx] = (0, 0, 255)
# Draw the edge (white) and the resulting ellipse (red)
edges = color.gray2rgb(img_as_ubyte(edges))
edges[cy, cx] = (250, 0, 0)

fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4),
                                subplot_kw={'adjustable':'box-forced'})

ax1.set_title('Original picture')
ax1.imshow(con)

ax2.set_title('Edge (white) and result (red)')
ax2.imshow(edges)

plt.show()
print("Tiempo:"+str(time.time()-start))
plt.figure()
plt.imshow(diff,cmap='gray')
plt.show()
exit()


its = 50
con=cv2.erode(con, None, iterations=its)
sin=cv2.erode(sin, None, iterations=its)

diff = sin*0.9 - con
plt.figure()
plt.title("Diferencia de erode")
plt.imshow(diff)
plt.show()
plt.figure()
plt.subplot(211)
plt.imshow(sin[:,:,::-1])
plt.subplot(212)
plt.imshow(con[:,:,::-1])

sin_gray = cv2.cvtColor(sin, cv2.COLOR_BGR2GRAY)
con_gray = cv2.cvtColor(con, cv2.COLOR_BGR2GRAY)
th = 80
trash,sin_BW = cv2.threshold(sin_gray, th, 255, cv2.THRESH_BINARY)
trash,con_BW = cv2.threshold(con_gray, th, 255, cv2.THRESH_BINARY)

sin_im2, sin_contours, sin_hierarchy = cv2.findContours(sin_BW,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
con_im2, con_contours, con__hierarchy = cv2.findContours(con_BW,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(sin, sin_contours, -1, (0,255,0), 20)
cv2.drawContours(con, con_contours, -1, (0,255,0), 20)
sin_empty = np.zeros(np.shape(sin))
con_empty = np.zeros(np.shape(sin))
cv2.drawContours(sin_empty, sin_contours, -1, (0,255,0), 3)
cv2.drawContours(con_empty, con_contours, -1, (0,255,0), 3)
plt.figure()
plt.subplot(211)
plt.title("Contornos")
plt.imshow(sin_empty[:,:,::-1])
plt.subplot(212)
plt.imshow(con_empty[:,:,::-1])

plt.show()

plt.figure()
plt.subplot(211)
plt.imshow(sin[:,:,::-1])
plt.subplot(212)
plt.imshow(con[:,:,::-1])
plt.show()
plt.figure()
plt.imshow(con-sin)
plt.show()
plt.figure()
plt.imshow(cv2.erode(sin[:,:,::-1], None, iterations=10)-cv2.erode(con[:,:,::-1], None, iterations=10))
plt.show()


kernel = np.ones((5,5), np.uint8)
sin = cv2.cvtColor(sin, cv2.COLOR_BGR2GRAY)
sin = cv2.bitwise_not(sin)
th = 150
trash,sin = cv2.threshold(sin, th, 255, cv2.THRESH_BINARY)

con = cv2.cvtColor(con, cv2.COLOR_BGR2GRAY)
con = cv2.bitwise_not(con)
trash,con = cv2.threshold(con, th, 255, cv2.THRESH_BINARY)
plt.figure()
its = 30

diff = cv2.dilate(cv2.erode(con, None, iterations=its)-cv2.erode(sin, None, iterations=its),kernel, iterations=its)
plt.imshow(diff,cmap = 'gray')



plt.show()

"""