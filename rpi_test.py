import numpy as np
import matplotlib.pylab as plt
import cnn
from keras.models import load_model
import cv2

model = load_model('model.h5')


# initialize the camera
cam = cv2.VideoCapture(0)   # 0 -> index of camera
s, img = cam.read()
if s:    # frame captured without any errors
    cv2.namedWindow("cam-test",cv2.WINDOW_NORMAL)
    cv2.imshow("cam-test",img)
    cv2.waitKey(0)
    cv2.destroyWindow("cam-test")
    #imwrite("filename.jpg",img) #save image