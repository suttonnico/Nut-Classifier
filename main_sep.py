
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


train_imgs,train_lbls,test_imgs,test_lbls = get_test_train_sep(0.8,dif=size)
my_cnn = cnn.cnn_sep(img_width=W, img_height=H)

batch_size = 200
epochs = 1000
# train
print(np.shape(train_imgs))
#exit()

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
history = my_cnn.fit(x=train_imgs,             # Input should be (train_cases, 128, 128, 1)
                     y=train_lbls,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=2,
                     validation_data=(test_imgs, test_lbls),callbacks=[tensorboard,es],
                     )
my_cnn.save('model_sep.h5')
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

#Confution Matrix and Classification Report
Y_pred = my_cnn.predict_classes(test_imgs,batch_size=int(len(test_imgs)/10))
y_pred = np.argmax(Y_pred, axis=1)
#print(Y_pred)
#print(y_pred)
print(test_lbls)
con_mat =  confusion_matrix(test_lbls, Y_pred)
print(con_mat)
figure = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat, annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()