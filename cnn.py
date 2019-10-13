from keras import layers, models, optimizers
import numpy as np

def cnn_sep(img_width=640, img_height=480):
    # https://www.kaggle.com/crawford/lung-infiltration-cnn-with-keras-on-chest-x-rays
    layer_C1 = 20
    layer_C2 = 20
    dense_layer = 200
    model = models.Sequential()
    model.add(layers.Conv2D(layer_C1, (3, 3), input_shape=(img_height,img_width,3),strides=1))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPool2D((3,3)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(layer_C2, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(dense_layer))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))

    model.add(layers.Dense(1))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("sigmoid"))

    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.Adam(lr=0.0001, beta_1=0.9999, beta_2=0.999, epsilon=None, decay=0.0),
        metrics=['acc'])

    model.summary()
    return model


def cnn(img_width=640, img_height=480):
    # https://www.kaggle.com/crawford/lung-infiltration-cnn-with-keras-on-chest-x-rays
    layer_C1 = 10
    layer_C2 = 20
    dense_layer = 200
    model = models.Sequential()
    model.add(layers.Conv2D(layer_C1, (5, 5), input_shape=(img_height,img_width*2,3),strides=3))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPool2D((3,3)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(layer_C2, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(dense_layer))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))

    model.add(layers.Dense(1))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("sigmoid"))

    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.Adam(lr=0.0001, beta_1=0.9999, beta_2=0.999, epsilon=None, decay=0.0),
        metrics=['acc'])

    model.summary()
    return model
