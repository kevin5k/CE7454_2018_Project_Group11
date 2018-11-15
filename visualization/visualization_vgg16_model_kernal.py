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

model_v4 = Sequential()

model_v4.add(ZeroPadding2D((1, 1), input_shape=(64, 64, 1)))
model_v4.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
model_v4.add(BatchNormalization())
model_v4.add(MaxPooling2D(pool_size=(2, 2)))
model_v4.add(Dropout(0.25))

model_v4.add(Convolution2D(128, (3, 3), activation='relu', padding='same', ))
model_v4.add(BatchNormalization())
model_v4.add(MaxPooling2D(pool_size=(2, 2)))
model_v4.add(Dropout(0.25))

model_v4.add(Convolution2D(256, (3, 3), activation='relu', padding='same', ))
model_v4.add(BatchNormalization())
model_v4.add(MaxPooling2D(pool_size=(2, 2)))
model_v4.add(Dropout(0.25))

model_v4.add(Convolution2D(512, (3, 3), activation='relu', padding='same', ))

def process(x):
    res = np.clip(x, 0, 1)
    return res

def dprocessed(x):
    res = np.zeros_like(x)
    res += 1
    res[x < 0] = 0
    res[x > 1] = 0
    return res

def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

for i_kernal in range(32):
    input_img=model_v4.input
    loss = K.mean(model_v4.layers[5].output[:, :,:,i_kernal])
    # loss = K.mean(model_v4.output[:, i_kernal])
    # compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]
    # normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img, K.learning_phase()], [loss, grads])
    # we start from a gray image with some noise
    np.random.seed(0)
    num_channels=1
    img_height=img_width=64
    input_img_data = (255- np.random.randint(0,255,(1,  img_height, img_width, num_channels))) / 255.
    failed = False
    # run gradient ascent
    if i_kernal%8 == 0:
        print('###    ', i_kernal+1)
    loss_value_pre=0
    for i in range(100):
        # processed = process(input_img_data)
        # predictions = model_v4.predict(input_img_data)
        loss_value, grads_value = iterate([input_img_data,1])
        # grads_value *= dprocessed(input_img_data[0])
        if i%1000 == 0:
            # print(' predictions: ' , np.shape(predictions), np.argmax(predictions))
            #print('Iteration %d/%d, loss: %f' % (i, 10000, loss_value))
            #print('Mean grad: %f' % np.mean(grads_value))
            if all(np.abs(grads_val) < 0.000001 for grads_val in grads_value.flatten()):
                failed = True
                print('Failed')
                break
            # print('Image:\n%s' % str(input_img_data[0,0,:,:]))
            if loss_value_pre != 0 and loss_value_pre > loss_value:
                break
            if loss_value_pre == 0:
                loss_value_pre = loss_value

        input_img_data += grads_value * 1 #e-3
    plt.subplot(4, 8, i_kernal+1)
    # plt.imshow((process(input_img_data[0,:,:,0])*255).astype('uint8'), cmap='Greys') #cmap='Greys'
    img_re = deprocess_image(input_img_data[0])
    img_re = np.reshape(img_re, (64,64))
    plt.imshow(img_re) #, cmap='Greys') #cmap='Greys'
plt.show()