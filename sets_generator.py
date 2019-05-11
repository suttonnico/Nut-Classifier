import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import imutils


greenLower = (0, 0, 0)
greenUpper = (255, 150, 150)

def lp(img):
    size = 21
    smallBlur = np.ones((size, size), dtype="float") * (1.0 / (size * size))
    convoleOutput = cv2.filter2D(img,-1, smallBlur)
    return convoleOutput

def get_nut(img,dif=150):
    plt.figure()
    plt.imshow(img)
    plt.show()

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
    plt.figure()
    plt.imshow(BW,cmap='gray')
    plt.show()
    laplacian = cv2.Laplacian(BW, cv2.CV_64F)
    plt.figure()
    plt.imshow(laplacian, cmap='gray')
    plt.show()
    #bausra, BW = cv2.threshold(laplacian, 100, 255, cv2.THRESH_BINARY)
    BW = lp(BW)
    H,W=np.shape(BW)
    plt.figure()
    plt.imshow(BW, cmap='gray')
    plt.show()
    B_H = []
    B_W = []
    for i in range(H):
        for j in range(W):
            if BW[i,j] == 0:
                B_H = np.append(B_H,i)
                B_W = np.append(B_W,j)
    m_B_H = int(np.mean(B_H))
    m_B_W = int(np.mean(B_W))
    #dif = 150
    if m_B_H + dif > H:
        low_limit = H
        up_limit = H -2*dif
    else:
        if m_B_H - dif < 0:
            up_limit = 0
            low_limit = 2*dif
        else:
            up_limit = m_B_H - dif
            low_limit = m_B_H + dif

    if m_B_W + dif > W:
        right_limit = W
        left_limit = W -2*dif
    else:
        if m_B_W - dif < 0:
            left_limit = 0
            right_limit= 2*dif
        else:
            left_limit = m_B_W - dif
            right_limit = m_B_W + dif
    plt.figure()
    plt.imshow(img[up_limit:low_limit,left_limit:right_limit,:])
    plt.show()
    return img[up_limit:low_limit,left_limit:right_limit,:]

def get_test_train(percentage,dif = 150):
    nut_dir = 'data/nuts_hd'

    labels = np.genfromtxt('data/labels_hd.csv', delimiter=',')
    imgs_files = [f for f in os.listdir(nut_dir)]

    images = []

    for f in imgs_files:
        print(f)
        img = cv2.imread(os.path.join(nut_dir, f))  # img.shape ~ (2919, 3000)
        get_nut(img)
        #img = get_nut(img)
        img = cv2.resize(img,(2*dif, 2*dif))
        images.append(img)
    inds = np.arange(len(labels))
    np.random.shuffle(inds)
    sh_images = []
    sh_labels = np.zeros(len(labels))
    for i in range(len(labels)):
        sh_labels[i] = labels[inds[i]]
        sh_images.append(images[inds[i]])
    train_imgs = []
    train_lbls = []
    test_imgs = []
    test_lbls = []
    for i in range(len(labels)):
        if i/len(labels) < percentage:
            train_imgs.append(sh_images[i])
            train_lbls.append(sh_labels[i])
        else:
            test_imgs.append(sh_images[i])
            test_lbls.append(sh_labels[i])
    return np.array(train_imgs),np.array(train_lbls),np.array(test_imgs),np.array(test_lbls)


