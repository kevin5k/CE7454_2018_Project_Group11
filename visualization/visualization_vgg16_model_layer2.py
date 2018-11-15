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

model_v2 = Sequential()

model_v2.add(ZeroPadding2D((1, 1), input_shape=(64, 64, 1)))
model_v2.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
model_v2.add(BatchNormalization())
model_v2.add(MaxPooling2D(pool_size=(2, 2)))
model_v2.add(Dropout(0.25))

model_v2.add(Convolution2D(128, (3, 3), activation='relu', padding='same', ))
print(len(model_v2.layers))
layer1 = model.layers[1]
weight1 = layer1.get_weights()
model_v2.layers[1].set_weights(weight1)
layer5 = model.layers[5]
weight5 = layer5.get_weights()
model_v2.layers[5].set_weights(weight5)
re2 = model_v2.predict(np.array(test_x))
re2 = np.transpose(re2, (0,3,1,2))
for i in range(32):
    plt.subplot(4,8,i+1)
    plt.imshow(re2[0][i]) #, cmap='gray'
plt.show()