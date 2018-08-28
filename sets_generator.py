import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

def lp(img):
    smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
    convoleOutput = cv2.filter2D(img,-1, smallBlur)
    return convoleOutput

def get_nut(img,dif=150):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = lp(gray)
    bausra, BW = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    laplacian = cv2.Laplacian(BW, cv2.CV_64F)
    #bausra, BW = cv2.threshold(laplacian, 100, 255, cv2.THRESH_BINARY)
    BW = lp(BW)
    H,W=np.shape(BW)
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



    return img[up_limit:low_limit,left_limit:right_limit,:]

def get_test_train(percentage,dif = 150):
    nut_dir = 'data/nuts'

    labels = np.genfromtxt('data/labels.csv', delimiter=',')
    imgs_files = [f for f in os.listdir(nut_dir)]

    images = []

    for f in imgs_files:
        print(f)
        img = cv2.imread(os.path.join(nut_dir, f))  # img.shape ~ (2919, 3000)
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


