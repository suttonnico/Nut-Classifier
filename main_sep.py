
from keras import layers, models, optimizers
import sklearn
import itertools
import numpy as np
import matplotlib.pylab as plt
import cnn
import seaborn as sns
from keras.callbacks import EarlyStopping
from sets_generator import get_test_train_sep
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import tensorflow as tf
from time import time
from sklearn.metrics import classification_report, confusion_matrix
from cm import plot_confusion_matrix



size = 150
W = 160
H = 120
batch_size = 200
epochs = 1000
id = '6'
train_imgs,train_lbls,test_imgs,test_lbls = get_test_train_sep(0.8,id,dif=size)
my_cnn = cnn.cnn_sep(img_width=W, img_height=H)
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
history = my_cnn.fit(x=train_imgs,y=train_lbls,batch_size=batch_size,epochs=epochs,verbose=2,validation_data=(test_imgs, test_lbls),callbacks=[tensorboard,es],)
my_cnn.save('model_sep_'+id+'.h5')
print(np.max(history.history['val_acc']))
acc_t_0=history.history['acc']
acc_v=history.history['val_acc']

