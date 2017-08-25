from keras.layers import Reshape, Flatten, Activation
from keras.layers.core import Dense
from keras.layers.convolutional import Conv3D, Conv3DTranspose, UpSampling3D
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras_adversarial.legacy import l1l2

def em_generator(latent_dim, input_shape, leaky_alpha = 6*[0.2],reg = lambda: l1l2(1e-7, 1e-7)):
    model = Sequential()

    model.add(Dense(1728, input_shape=(latent_dim,), kernel_regularizer=reg()))
    model.add(LeakyReLU(leaky_alpha[0]))
    model.add(Reshape([6,6,3,16]))
    model.add(UpSampling3D((2,2,2)))

    model.add(Conv3DTranspose(64, (5,5,3), kernel_regularizer=reg()))
    model.add(LeakyReLU(leaky_alpha[1]))

    model.add(Conv3DTranspose(32, (5,5,3), kernel_regularizer=reg()))
    model.add(LeakyReLU(leaky_alpha[2]))

    model.add(Conv3DTranspose(16, (5,5,3), kernel_regularizer=reg()))
    model.add(LeakyReLU(leaky_alpha[3]))

    model.add(Conv3DTranspose(16, (3,3,3), kernel_regularizer=reg()))
    model.add(LeakyReLU(leaky_alpha[4]))

    model.add(Conv3D(8, (3,3,3), kernel_regularizer=reg()))
    model.add(LeakyReLU(leaky_alpha[5]))

    model.add(Conv3D(1, (1,1,1), activation="sigmoid", kernel_regularizer=reg()))
    return model

def em_discriminator(input_shape, leaky_alpha = 5*[0.2], reg = lambda: l1l2(1e-7, 1e-7)):
    disc = Sequential()

    disc.add(Conv3D(128, (5,5,3), input_shape=(input_shape+(1,)), kernel_regularizer=reg()))
    disc.add(LeakyReLU(leaky_alpha[0]))

    disc.add(Conv3D(64, (3,3,3), kernel_regularizer=reg()))
    disc.add(LeakyReLU(leaky_alpha[1]))

    disc.add(Conv3D(32, (3,3,3), kernel_regularizer=reg()))
    disc.add(LeakyReLU(leaky_alpha[2]))

    disc.add(Conv3D(8, (1,1,1), kernel_regularizer=reg()))
    disc.add(LeakyReLU(leaky_alpha[3]))

    disc.add(Flatten())
    disc.add(Dense(8, kernel_regularizer=reg()))
    disc.add(LeakyReLU(leaky_alpha[4]))

    disc.add(Dense(1))
    disc.add(Activation("sigmoid"))
    return disc



def em_generator_large(latent_dim, input_shape, leaky_alpha = 7*[0.2], reg = lambda: l1l2(1e-7, 1e-7)):
    model = Sequential()

    model.add(Dense(3072, input_shape=(latent_dim,), kernel_regularizer=reg()))
    model.add(LeakyReLU(leaky_alpha[0]))
    model.add(Reshape([8,8,3,16]))
    model.add(UpSampling3D((6,6,2)))

    model.add(Conv3DTranspose(64, (7,7,3), kernel_regularizer=reg()))
    model.add(LeakyReLU(leaky_alpha[1]))

    model.add(Conv3DTranspose(32, (7,7,3), kernel_regularizer=reg()))
    model.add(LeakyReLU(leaky_alpha[2]))

    model.add(Conv3DTranspose(16, (5,5,3), kernel_regularizer=reg()))
    model.add(LeakyReLU(leaky_alpha[3]))

    model.add(Conv3DTranspose(16, (5,5,3), kernel_regularizer=reg()))
    model.add(LeakyReLU(leaky_alpha[4]))

    model.add(Conv3D(8, (3,3,5), kernel_regularizer=reg()))
    model.add(LeakyReLU(leaky_alpha[5]))

    model.add(Conv3D(8, (3,3,4), kernel_regularizer=reg()))
    model.add(LeakyReLU(leaky_alpha[6]))

    model.add(Conv3D(1, (1,1,1), activation="sigmoid", kernel_regularizer=reg()))
    return model

def em_discriminator_large(input_shape, leaky_alpha = 7*[0.2], reg = lambda: l1l2(1e-7, 1e-7)):
    disc = Sequential()

    disc.add(UpSampling3D((1,1,2), input_shape=(input_shape+(1,))))

    disc.add(Conv3D(128, (7,7,3), kernel_regularizer=reg()))
    disc.add(LeakyReLU(leaky_alpha[0]))

    disc.add(Conv3D(64, (5,5,3), kernel_regularizer=reg()))
    disc.add(LeakyReLU(leaky_alpha[1]))

    disc.add(Conv3D(64, (5,5,3), kernel_regularizer=reg()))
    disc.add(LeakyReLU(leaky_alpha[2]))

    disc.add(Conv3D(64, (3,3,3), kernel_regularizer=reg()))
    disc.add(LeakyReLU(leaky_alpha[3]))

    disc.add(Conv3D(32, (3,3,1), kernel_regularizer=reg()))
    disc.add(LeakyReLU(leaky_alpha[4]))

    disc.add(Conv3D(16, (1,1,1), kernel_regularizer=reg()))
    disc.add(LeakyReLU(leaky_alpha[5]))

    disc.add(Flatten())
    disc.add(Dense(16, kernel_regularizer=reg()))
    disc.add(LeakyReLU(leaky_alpha[6]))

    disc.add(Dense(1))
    disc.add(Activation("sigmoid"))
    return disc
