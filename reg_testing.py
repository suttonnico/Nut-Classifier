import cv2
import cnn
from keras.models import load_model
import os
import numpy as np

size = 150
W = 2*size
H = 2*size

def zero_pad(x,n):
    for i in range(1,5):
        if x < 10 ** i:
            return (n-i)*'0'+str(x)

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

model = load_model('model.h5')
weights = model.get_weights()
my_cnn = cnn.cnn(img_width=W, img_height=H)
my_cnn.set_weights(weights)

labels = np.genfromtxt('data_cinta/dataset/labels.csv', delimiter=',')

nut_dir_sep = 'data_cinta/good_11'

pairs = {
    '0': '6',
    '2': '4'
}

new_imgs_files = [f for f in os.listdir(nut_dir_sep)]
out = []

for f in new_imgs_files:
    if f != 'labels.csv':
        id = getNutId(f)
        num = getNutNumber(f)
        if id in pairs:
            pair_id = pairs[id]
            img_org = cv2.imread(os.path.join(nut_dir_sep, f))
            img_pair = cv2.imread(os.path.join(nut_dir_sep, subNutId(f, pair_id)))
            img = np.concatenate((img_org, img_pair), axis=1)
            img = cv2.resize(img, (4 * size, 2 * size))
            pred = model.predict_classes(img.reshape([-1, 300, 600, 3]), batch_size=1)
            out.append(pred)
            print(f+" :" + str(pred))

