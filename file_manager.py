import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

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

#DO NOT RUN

if __name__ == '__main__':

    nut_dir = 'data_posta/dataset'
    nut_dir_sep = 'data_posta/dataset_sep'
    imgs_files = [f for f in os.listdir(nut_dir)]
    last_nut = 0
    for f in imgs_files:
        last_nut+=1
    new_nut_dir = 'data_posta/good_04'
    good_or_bad = 0 #1= mala 0 = nuez buena

    labels = np.genfromtxt('data_posta/dataset/labels.csv', delimiter=',')


    pairs = {
        '0':'6',
        '2':'4'
    }

    new_imgs_files = [f for f in os.listdir(new_nut_dir)]
    for f in new_imgs_files:
        id = getNutId(f)
        num = getNutNumber(f)
        print("ID: "+id)
        print("NUM: "+num)
        if id in pairs:
            pair_id = pairs[id]
            img_org = cv2.imread(os.path.join(new_nut_dir, f))
            img_pair = cv2.imread(os.path.join(new_nut_dir, subNutId(f,pair_id)))
            img = np.concatenate((img_org, img_pair), axis=1)
            cv2.imwrite(nut_dir_sep + '/nuez'+id+'_' + zero_pad(last_nut, 6) + '.png', img_org);
            cv2.imwrite(nut_dir_sep + '/nuez'+pair_id+'_' + zero_pad(last_nut, 6) + '.png', img_pair);
            cv2.imwrite(nut_dir+'/nuez_'+zero_pad(last_nut,6)+'.png', img);
            last_nut += 1
            labels=np.append(labels,int(good_or_bad))

    print(labels)
    np.savetxt('data_posta/dataset/labels.csv', labels, delimiter=",")
    np.savetxt('data_posta/dataset_sep/labels.csv', labels, delimiter=",")
