import numpy as np
import os
import sys
from PIL import Image
from keras.layers import Reshape, Flatten, LeakyReLU, Activation
from keras.layers.core import Dense, Dropout
from keras.layers.convolutional import MaxPooling3D, Conv3D
from keras.models import Sequential
from keras.optimizers import Adam, SGD


#placeholder convolutional net
def train_em_gan(epochs, learning_rate):
    model = Sequential()
    model.add(Conv3D(32, (5,5,3), input_shape=(24,24,12,1), activation="relu"))
    model.add(MaxPooling3D((2,2,2)))
    model.add(Conv3D(16, (3,3,3), input_shape=(10,10,5,32), activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    print(model.summary())

if __name__=="__main__":
    train_em_gan(1,0.001)