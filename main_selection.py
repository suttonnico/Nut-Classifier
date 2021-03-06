from keras import layers, models, optimizers
import sklearn
import itertools
import numpy as np
import matplotlib.pylab as plt
import cnn
from sets_generator import get_test_train_sel
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from time import time
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
size = 20
W = 2*size
H = 2*size


train_imgs,train_lbls,test_imgs,test_lbls = get_test_train_sel(0.7,dif=size)
my_cnn = cnn.cnn_sel(img_width=W, img_height=H)

batch_size = 2
epochs = 500
# train
print(np.shape(train_imgs))
#exit()

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

history = my_cnn.fit(x=train_imgs,             # Input should be (train_cases, 128, 128, 1)
                     y=train_lbls,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=2,
                     validation_data=(test_imgs, test_lbls),callbacks=[tensorboard]
                     )
my_cnn.save('model_sel.h5')
print(np.max(history.history['val_acc']))
plt.figure()
plt.plot(history.history['acc'],label='train accuracy')
plt.plot(history.history['val_acc'],label='test accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(loc='lower right')

plt.figure()
plt.plot(history.history['loss'],label='train accuracy')
plt.plot(history.history['val_loss'],label='test accuracy')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(loc='lower right')
plt.show()