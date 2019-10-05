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
    exit(11111)
    nut_dir = 'data_cinta/dataset'
    nut_dir_sep = 'data_cinta/dataset_sep'
    imgs_files = [f for f in os.listdir(nut_dir)]
    last_nut = 0
    for f in imgs_files:
        last_nut+=1
    new_nut_dir_arr = ['data_cinta/bad_03','data_cinta/bad_02','data_cinta/bad_01','data_cinta/good_01','data_cinta/good__02','data_cinta/good_03']
    good_or_bad_arr = [1,1,1,0,0,0] #1= mala 0 = nuez buena
    labels = []  # np.genfromtxt('data_cinta/dataset/labels.csv', delimiter=',')

    for i in range(len(new_nut_dir_arr)):
        new_nut_dir=new_nut_dir_arr[i]
        good_or_bad=good_or_bad_arr[i]


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
                cv2.imwrite(nut_dir_sep + '/nuez'+ zero_pad(last_nut, 6) +'_' +id+ '.png', img_org);
                cv2.imwrite(nut_dir_sep + '/nuez'+ zero_pad(last_nut, 6) +'_' +pair_id+ '.png', img_pair);
                cv2.imwrite(nut_dir+'/nuez_'+zero_pad(last_nut,6)+'.png', img);
                last_nut += 1
                labels=np.append(labels,int(good_or_bad))

        print(labels)
    np.savetxt('data_cinta/dataset/labels.csv', labels, delimiter=",")
    np.savetxt('data_cinta/dataset_sep/labels.csv', labels, delimiter=",")
