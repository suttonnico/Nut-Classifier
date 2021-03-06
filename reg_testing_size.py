import cv2
import cnn
from keras.models import load_model
import os
import numpy as np
import size_classification
import matplotlib.pyplot as plt
size = 150
W = 2*size
H = 2*size

nut_dir_sep = 'data_cinta/good_13'

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


pairs = {
    '0':'6',
    '2':'4'
}

def getSide(x):
    return x[4]
new_imgs_files = [f for f in os.listdir(nut_dir_sep)]
out = []

empty_0 = cv2.imread(os.path.join(nut_dir_sep, 'empty0.png'))
empty_2 = cv2.imread(os.path.join(nut_dir_sep, 'empty2.png'))
empty_4 = cv2.imread(os.path.join(nut_dir_sep, 'empty4.png'))
empty_6 = cv2.imread(os.path.join(nut_dir_sep, 'empty6.png'))
test_id ='a000085'
for f in new_imgs_files:
    if f != 'labels.csv' and f!= 'empty0.png' and f!= 'empty2.png' and f!= 'empty4.png' and f!= 'empty6.png':
        id = getNutId(f)
        num = getNutNumber(f)

        print(num)
        if id == '0':
            pair_id = pairs[id]
            img_org = cv2.imread(os.path.join(nut_dir_sep, f))
            img_pair = cv2.imread(os.path.join(nut_dir_sep, subNutId(f, pair_id)))
            size1 = size_classification.findRadius(img_org, empty_0)
            size2 = size_classification.findRadius(img_pair, empty_6)
            diametro = round(size_classification.sizes2rad(size1, size2, 120), 2)
            if num == test_id:
                plt.figure()
                plt.subplot(121)
                plt.imshow(img_org)
                plt.subplot(122)
                plt.imshow(img_pair)
                plt.show()
            print("Diametro: " + str(diametro))


            out.append(diametro)
        if id == '2':
            pair_id = pairs[id]
            img_org = cv2.imread(os.path.join(nut_dir_sep, f))
            img_pair = cv2.imread(os.path.join(nut_dir_sep, subNutId(f, pair_id)))

            size1 = size_classification.findRadius(img_org, empty_2)
            size2 = size_classification.findRadius(img_pair, empty_4)
            diametro = round(size_classification.sizes2rad(size1, size2, 120), 2)
            print("Diametro: " + str(diametro))

            out.append(diametro)


print(np.mean(out))
print(np.std(out))

plt.figure()
plt.hist(out,bins=10)
plt.title("Histograma de mediciones calibre")
plt.show()