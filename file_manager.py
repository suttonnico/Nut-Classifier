import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def zero_pad(x,n):
    for i in range(1,5):
        if x < 10 ** i:
            return (n-i)*'0'+str(x)


#DO NOT RUN
nut_dir = 'data/nuts_hd'
imgs_files = [f for f in os.listdir(nut_dir)]
last_nut = 0
for f in imgs_files:
    last_nut+=1
new_nut_dir = 'nuez_mala'
good_or_bad = 1

labels = np.genfromtxt('data/labels_hd.csv', delimiter=',')

new_imgs_files = [f for f in os.listdir(new_nut_dir)]
for f in new_imgs_files:
    img = cv2.imread(os.path.join(new_nut_dir, f))
    cv2.imwrite(nut_dir+'/nuez_'+zero_pad(last_nut,4)+'.png', img);
    last_nut += 1
    labels=np.append(labels,int(good_or_bad))

#print(labels)
np.savetxt("data/labels_hd.csv", labels, delimiter=",")
