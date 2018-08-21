import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

def get_nut(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bausra, BW = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    laplacian = cv2.Laplacian(BW, cv2.CV_64F)

    plt.imshow(laplacian,cmap='gray')
    plt.show()

def get_test_train(percentage,img_width,img_height):
    nut_dir = 'data/nuts'

    labels = np.genfromtxt('data/labels.csv', delimiter=',')
    imgs_files = [f for f in os.listdir(nut_dir)]

    images = []
    for f in imgs_files:
        # print(f)
        #print(f)
        img = cv2.imread(os.path.join(nut_dir, f))  # img.shape ~ (2919, 3000)
        #get_nut(img)
        img = cv2.resize(img,(img_width, img_height))
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


