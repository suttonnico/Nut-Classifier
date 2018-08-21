from keras import layers, models, optimizers
import sklearn
import itertools
import numpy as np
import matplotlib.pylab as plt
import cnn
from sets_generator import get_test_train
from keras.callbacks import ModelCheckpoint

W = 200
H = 150


train_imgs,train_lbls,test_imgs,test_lbls = get_test_train(0.5,img_width=W, img_height=H)
my_cnn = cnn.cnn(img_width=W, img_height=H)

batch_size = 50
epochs = 200
# train
print(np.shape(train_imgs))
#exit()
history = my_cnn.fit(x=train_imgs,             # Input should be (train_cases, 128, 128, 1)
                     y=train_lbls,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=2,
                     validation_data=(test_imgs, test_lbls)
                     )

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