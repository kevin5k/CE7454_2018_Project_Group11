#  libraries
from keras.layers import Convolution2D, MaxPooling2D, Activation, ZeroPadding2D, BatchNormalization, Dropout, AveragePooling2D, Flatten, Dense
from keras.models import Sequential
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2

# read image
ship = cv2.imread('vis_image\ship.jpg')
plt.imshow(ship)
ship.shape

#visualise image
def visualize(ship_batch):
    conv_ship = np.squeeze(ship_batch, axis = 0)
    print(conv_ship.shape)
    conv_ship = (255 - conv_ship) / 255
    plt.imshow(conv_ship)

# one layer model
#     3 x 3 kernal
model_l1_k3x3 = Sequential()
model_l1_k3x3.add(Convolution2D(3, (3, 3), input_shape = ship.shape))
ship_batch = np.expand_dims(ship, axis = 0)
conv_ship = model_l1_k3x3.predict(ship_batch)
visualize(conv_ship)

#     4 x 4 kernal
model_l1_k4x4 = Sequential()
model_l1_k4x4.add(Convolution2D(3, (4, 4), input_shape = ship.shape))
ship_batch = np.expand_dims(ship, axis = 0)
conv_ship = model_l1_k4x4.predict(ship_batch)
visualize(conv_ship)

#     6 x 6 kernal
model_l1_k6x6 = Sequential()
model_l1_k6x6.add(Convolution2D(3, (6, 6), input_shape = ship.shape))
ship_batch = np.expand_dims(ship, axis = 0)
conv_ship = model_l1_k6x6.predict(ship_batch)
visualize(conv_ship)

#     3 filter layers
model_l1_f03 = Sequential()
model_l1_f03.add(Convolution2D(3, (4, 4), input_shape = ship.shape))
ship_batch = np.expand_dims(ship, axis = 0)
conv_ship = model_l1_f03.predict(ship_batch)
visualize(conv_ship)

#     4 filter layers
model_l1_f04 = Sequential()
model_l1_f04.add(Convolution2D(4, (4, 4), input_shape = ship.shape))
ship_batch = np.expand_dims(ship, axis = 0)
conv_ship = model_l1_f04.predict(ship_batch)
visualize(conv_ship)

# Model: One convolutional layer with one activation layer
activation_layer_model = Sequential()
activation_layer_model.add(Convolution2D(3, (3, 3), input_shape = ship.shape))
activation_layer_model.add(Activation('relu'))
ship_batch = np.expand_dims(ship, axis = 0)
conv_ship = activation_layer_model.predict(ship_batch)
visualize(conv_ship)

# Model: One convolutional layer with one activation layer and one pooling layer, size of (2, 2)
pooling_layer_model = Sequential()
pooling_layer_model.add(Convolution2D(3, (3, 3), input_shape = ship.shape))
pooling_layer_model.add(Activation('relu'))
pooling_layer_model.add(MaxPooling2D(pool_size = (2, 2)))
ship_batch = np.expand_dims(ship, axis = 0)
conv_ship = pooling_layer_model.predict(ship_batch)
visualize(conv_ship)

# Model: One convolutional layer with one activation layer and one pooling layer, size of (2, 2)
pooling_layer_model = Sequential()
pooling_layer_model.add(Convolution2D(3, (3, 3), input_shape = ship.shape))
pooling_layer_model.add(Activation('relu'))
pooling_layer_model.add(MaxPooling2D(pool_size = (4, 4)))
ship_batch = np.expand_dims(ship, axis = 0)
conv_ship = pooling_layer_model.predict(ship_batch)
visualize(conv_ship)

# Model: One convolutional layer with one activation layer and one pooling layer, size of (2, 2)
pooling_layer_model = Sequential()
pooling_layer_model.add(Convolution2D(3, (3, 3), input_shape = ship.shape))
pooling_layer_model.add(Activation('relu'))
pooling_layer_model.add(MaxPooling2D(pool_size = (8, 8)))
ship_batch = np.expand_dims(ship, axis = 0)
conv_ship = pooling_layer_model.predict(ship_batch)
visualize(conv_ship)