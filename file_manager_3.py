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
    nut_dir_0 = 'data_cinta/dataset_0'
    nut_dir_2 = 'data_cinta/dataset_2'
    nut_dir_4 = 'data_cinta/dataset_4'
    nut_dir_6 = 'data_cinta/dataset_6'


    os.mkdir(nut_dir_0, 777)
    os.mkdir(nut_dir_2, 777)
    os.mkdir(nut_dir_4, 777)
    os.mkdir(nut_dir_6, 777)
    last_nut_0 = 0
    last_nut_2 = 0
    last_nut_4 = 0
    last_nut_6 = 0

    new_nut_dir_arr = ['data_cinta/bad_09',
                       'data_cinta/bad_08',
                       'data_cinta/bad_07',
                       'data_cinta/good_10',
                       'data_cinta/good_09',
                       'data_cinta/bad_06',
                       'data_cinta/good_07',
                       'data_cinta/good_06',
                       'data_cinta/good_05',
                       'data_cinta/good_04',
                       'data_cinta/bad_05',
                       'data_cinta/bad_04',
                       'data_cinta/bad_03',
                       'data_cinta/bad_02',
                       'data_cinta/bad_01',
                       'data_cinta/good_01',
                       'data_cinta/good__02',
                       'data_cinta/good_03',
                       'data_cinta/good_11']
    good_or_bad_arr = [1,1,1,0,0,1,0,0,0,0,1,1,1,1,1,0,0,0,0] #1= mala 0 = nuez buena
    labels_0 = []  # np.genfromtxt('data_cinta/dataset/labels.csv', delimiter=',')
    labels_2 = []
    labels_4 = []
    labels_6 = []
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
            if id == '0':
                print(id)
                img = cv2.imread(os.path.join(new_nut_dir, f))
                cv2.imwrite(nut_dir_0 + '/nuez' + zero_pad(last_nut_0, 6) + '_' + id + '.png', img)
                last_nut_0 += 1
                labels_0 = np.append(labels_0, int(good_or_bad))
            if id == '2':
                print(id)
                img = cv2.imread(os.path.join(new_nut_dir, f))
                cv2.imwrite(nut_dir_2 + '/nuez' + zero_pad(last_nut_2, 6) + '_' + id + '.png', img)
                last_nut_2 += 1
                labels_2 = np.append(labels_2, int(good_or_bad))
            if id == '4':
                print(id)
                img = cv2.imread(os.path.join(new_nut_dir, f))
                cv2.imwrite(nut_dir_4 + '/nuez' + zero_pad(last_nut_4, 6) + '_' + id + '.png', img)
                last_nut_4 += 1
                labels_4 = np.append(labels_4, int(good_or_bad))
            if id == '6':
                print(id)
                img = cv2.imread(os.path.join(new_nut_dir, f))
                cv2.imwrite(nut_dir_6 + '/nuez' + zero_pad(last_nut_6, 6) + '_' + id + '.png', img)
                last_nut_6 += 1
                labels_6 = np.append(labels_6, int(good_or_bad))

    np.savetxt('data_cinta/dataset_0/labels.csv', labels_0, delimiter=",")
    np.savetxt('data_cinta/dataset_2/labels.csv', labels_2, delimiter=",")
    np.savetxt('data_cinta/dataset_4/labels.csv', labels_4, delimiter=",")
    np.savetxt('data_cinta/dataset_6/labels.csv', labels_6, delimiter=",")

