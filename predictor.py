import cv2
import cnn
from keras.models import load_model
import os
import numpy as np

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

W = 160
H = 120

class predictor:
    model_1 = {}
    model_2 = {}

    def __init__(self,model_1,model_2):
        self.model_1 = model_1
        self.model_2 = model_2

    def predict(self,img1,img2):
        p_1 = self.model_1.predict_proba(img1, 1, 0)
        p_2 = self.model_2.predict_proba(img2, 1, 0)
        if (p_1 + p_2)> 1:
            return 1
        else:
            return 0

nut_dir_sep = 'data_cinta/good_reg'

pairs = {
    '0': '6',
    '2': '4'
}

new_imgs_files = [f for f in os.listdir(nut_dir_sep)]
out = []
weights = load_model('model_sep_0.h5').get_weights()
my_cnn_1 = cnn.cnn_sep(img_width=W, img_height=H)
my_cnn_1.set_weights(weights)
weights = load_model('model_sep_6.h5').get_weights()
my_cnn_2 = cnn.cnn_sep(img_width=W, img_height=H)
my_cnn_2.set_weights(weights)


pred_0_6 = predictor(my_cnn_1,my_cnn_2)
weights = load_model('model_sep_2.h5').get_weights()
my_cnn_1 = cnn.cnn_sep(img_width=W, img_height=H)
my_cnn_1.set_weights(weights)
weights = load_model('model_sep_4.h5').get_weights()
my_cnn_2 = cnn.cnn_sep(img_width=W, img_height=H)
my_cnn_2.set_weights(weights)

pred_2_4 = predictor(my_cnn_1,my_cnn_2)



out = []
p=0
for f in new_imgs_files:
    if f != 'labels.csv':
        id = getNutId(f)
        num = getNutNumber(f)
        if id == '0':
            pair_id = pairs[id]
            img_org = cv2.imread(os.path.join(nut_dir_sep, f))
            img_pair = cv2.imread(os.path.join(nut_dir_sep, subNutId(f, pair_id)))
            #predic = pred.predict(img_org,img_org)
            p=pred_0_6.predict(img_org.reshape([-1, 120, 160, 3]),img_pair.reshape([-1, 120, 160, 3]))
            out.append(p)
            print(f+" :" + str(p))
        if id == '2':
            pair_id = pairs[id]
            img_org = cv2.imread(os.path.join(nut_dir_sep, f))
            img_pair = cv2.imread(os.path.join(nut_dir_sep, subNutId(f, pair_id)))
            #predic = pred.predict(img_org,img_org)
            p=pred_2_4.predict(img_org.reshape([-1, 120, 160, 3]),img_pair.reshape([-1, 120, 160, 3]))
            print(f+" :" + str(p))
            out.append(p)

print("Performance = "+str(sum(out)/len(out)))
print("buenas: "+str(len(out)-sum(out))+" totales: "+str(len(out)))

