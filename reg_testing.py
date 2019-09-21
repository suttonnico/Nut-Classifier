import cv2
import cnn
from keras.models import load_model



model = load_model('model.h5')
weights = model.get_weights()
my_cnn = cnn.cnn(img_width=W, img_height=H)
my_cnn.set_weights(weights)


