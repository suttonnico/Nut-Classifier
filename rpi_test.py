import numpy as np
import cnn
from keras.models import load_model
import cv2
import time
size = 150
W = 2*size
H = 2*size

model = load_model('model.h5')
weights = model.get_weights()
my_cnn = cnn.cnn(img_width=W, img_height=H)

my_cnn.set_weights(weights)
dif = 150
# initialize the camera
cam = cv2.VideoCapture(0)   # 0 -> index of camera
N = 1000
start = time.time()
for i in range(N):
    s, img = cam.read()
    if s:    # frame captured without any errors
        my_cnn.predict_classes(cv2.resize(img,(2*dif, 2*dif)).reshape([-1,300,300,3]),batch_size=1)
        print(i)
    #imwrite("filename.jpg",img) #save image
end = time.time()
print((end-start)/N)

