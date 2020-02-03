import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import imutils
import file_manager as fm
import size_classification
kernel = np.ones((5,5), np.uint8)


def getNutId(x):
    return(x[11])

def subNutId(x,id):
    s = list(x)
    s[11] = id
    return "".join(s)

def getNutNumber(x):
    return(x[4:len(x)-6])


pairs = {
    '0':'6',
    '2':'4'
}

def getSide(x):
    return x[4]

greenLower = (0, 0, 0)
greenUpper = (255, 150, 150)

def lp(img):
    size = 21
    smallBlur = np.ones((size, size), dtype="float") * (1.0 / (size * size))
    convoleOutput = cv2.filter2D(img,-1, smallBlur)
    return convoleOutput


def get_nut(img, id, dif=150):
    empty = cv2.imread('data_cinta/empty/empty' + id + '.png')
    plt.figure()
    plt.imshow(img)
    plt.show()
    plt.figure()
    plt.imshow(empty)
    plt.show()

    th = 140

    empty_gray = cv2.cvtColor(empty, cv2.COLOR_BGR2GRAY)
    trash, empty_BW = cv2.threshold(empty_gray, th, 255, cv2.THRESH_BINARY)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    trash, img_BW = cv2.threshold(img_gray, th, 255, cv2.THRESH_BINARY)
    its = 50
    diff = cv2.dilate(cv2.erode(img, None, iterations=its) - cv2.erode(empty, None, iterations=its), kernel,
                      iterations=its)
    plt.imshow(diff, cmap='gray')

    #diff = empty_BW-img_BW
    plt.figure()
    plt.imshow(img_BW)
    plt.show()
    plt.figure()
    plt.imshow(empty_BW)
    plt.show()

    plt.figure()
    plt.imshow(diff)
    plt.show()


def get_nut_old(img,id,dif=150):
    empty=cv2.imread('data/empty/empty'+id+'.png')
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
    nut_dir = 'data_cinta/dataset_sep'

    labels = np.genfromtxt('data_cinta/dataset_sep/labels.csv', delimiter=',')
    imgs_files = [f for f in os.listdir(nut_dir)]
    print(labels)
    print("bad: "+str(sum(labels))+" good: "+str(len(labels)-sum(labels)))
    images = []
    images_size = []
    sizes = []
    i = 0
    for f in imgs_files:
        if f != 'labels.csv':

            id = getNutId(f)
            num = getNutNumber(f)

            if id in pairs:
                img_org = cv2.imread(os.path.join(nut_dir, f))  # img.shape ~ (2919, 3000)
                empty = cv2.imread('data_cinta/empty/empty' + id + '.png')
                img_pair = cv2.imread(os.path.join(nut_dir, subNutId(f,pairs[id])))
                empty_pair = cv2.imread('data_cinta/empty/empty' + pairs[id] + '.png')
                img = np.concatenate((img_org, img_pair), axis=1)
                img = cv2.resize(img, (4 * dif, 2 * dif))

                if labels[i] < 0.99:
                   #test.findRadius(img_org, empty)
                    #test.findRadius(img_pair, empty_pair)
                    #print(i)
                    #print(f,labels[i+1])
                    size1 = size_classification.findRadius(img_org, empty)
                    size2 = size_classification.findRadius(img_pair, empty_pair)
                    #print("pixeles camara 1:" + str(size1))
                    #print("pixeles camara 2:" + str(size2))
                    size = size_classification.sizes2rad(size1, size2, 120)
                    sizes.append(size)
                    images_size.append(img)
                    #print("Diametro: " + str(size))
                i += 1
                #get_nut(img_org,id)
            #img = get_nut(img)

               # plt.figure()
               # plt.imshow(img[:,:,::-1])
               # plt.show()
                images.append(img)
    dif =    int(sum(labels)-len(labels)+sum(labels))    #malas menos buenas
    if dif>0:
        imeges=images[dif:]
        labels=labels[dif:]
    print("bad: " + str(sum(labels)) + " good: " + str(len(labels) - sum(labels)))
    print("mean:" +str(np.mean(sizes)))
    print("dev:" + str(np.std(sizes)))
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

def get_test_train_sep(percentage,id,dif = 150):
    nut_dir = 'data_cinta/dataset_'+id

    labels = np.genfromtxt('data_cinta/dataset_'+id+'/labels.csv', delimiter=',')
    imgs_files = [f for f in os.listdir(nut_dir)]
    print(labels)
    print("bad: "+str(sum(labels))+" good: "+str(len(labels)-sum(labels)))
    images = []
    images_size = []
    sizes = []
    i = 0
    for f in imgs_files:
        if f != 'labels.csv':
            id = getNutId(f)
            num = getNutNumber(f)
            img = cv2.imread(os.path.join(nut_dir, f))  # img.shape ~ (2919, 3000)
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
