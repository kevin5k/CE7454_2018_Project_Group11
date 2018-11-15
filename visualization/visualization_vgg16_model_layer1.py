#  libraries
from keras.layers import Convolution2D, MaxPooling2D, Activation, ZeroPadding2D, BatchNormalization, Dropout, AveragePooling2D, Flatten, Dense
from keras.models import Sequential
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2

test_x = []
ship = cv2.imdecode(np.fromfile(r'vis_image\ship.jpg', dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
ship = cv2.resize(ship, (64, 64), interpolation=cv2.INTER_CUBIC)

ship = (255 - ship) / 255
ship = np.reshape(ship, (64, 64, 1))
test_x.append(ship)

###################################################################
layer = model.layers[1]
weight = layer.get_weights()
print(np.asarray(weight).shape)
###################################################################

model_v1 = Sequential()

model_v1.add(ZeroPadding2D((1, 1), input_shape=(64, 64, 1)))
model_v1.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
model_v1.layers[1].set_weights(weight)

re = model_v1.predict(np.array(test_x))
print(np.shape(re))
re = np.transpose(re, (0,3,1,2))
for i in range(32):
    plt.subplot(4,8,i+1)
    plt.imshow(re[0][i]) #, cmap='gray'
plt.show()