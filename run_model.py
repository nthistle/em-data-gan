import numpy as np
import os
import sys
from PIL import Image
from keras.layers import Reshape, Flatten, LeakyReLU, Activation
from keras.layers.core import Dense, Dropout
from keras.layers.convolutional import MaxPooling3D, Conv3D, Conv3DTranspose, UpSampling3D
from keras.models import Sequential
from keras_adversarial import AdversarialModel, simple_gan
from keras_adversarial import normal_latent_sampling, AdversarialOptimizerSimultaneous
from keras.callbacks import Callback
import util
from keras.optimizers import Adam


# class debugprint(Callback):
#
#     def on_batch_begin(self, batch, logs=None):
#         print("batch is beginning")
#         print(logs)
#
#     def on_batch_end(self, batch, logs=None):
#         print("batch is ending")
#         print(logs)


def em_generator(latent_dim, input_shape):
    model = Sequential()
    #model.add(Dense(784, input_shape=(latent_dim,), activation="relu"))
    model.add(Dense(1728, input_shape=(latent_dim,), activation="relu"))
    model.add(Reshape([6,6,3,16]))
    model.add(UpSampling3D((2,2,2)))
    model.add(Conv3DTranspose(64, (5,5,3), activation="relu"))
    model.add(Conv3DTranspose(32, (5,5,3), activation="relu"))
    model.add(Conv3DTranspose(16, (5,5,3), activation="relu"))
    model.add(Conv3DTranspose(16, (3,3,3), activation="relu"))
    model.add(Conv3D(8, (3,3,3), activation="relu"))
    model.add(Conv3D(1, (1,1,1), activation="sigmoid"))
    return model

def em_discriminator(input_shape):
    disc = Sequential()
    disc.add(Conv3D(128, (5,5,3), input_shape=(input_shape+(1,)), activation="relu"))
    #disc.add(Dropout(0.2))
    disc.add(Conv3D(64, (3,3,3), activation="relu"))
    #disc.add(Dropout(0.2))
    disc.add(Conv3D(32, (3,3,3), activation="relu"))
    disc.add(Conv3D(8, (1,1,1), activation="relu"))
    disc.add(Flatten())
    disc.add(Dense(8))
    disc.add(Activation("relu"))
    disc.add(Dense(1))
    disc.add(Activation("sigmoid"))
    return disc

def train_em_gan(adversarial_optimizer,
                 generator, discriminator, gen_opt, disc_opt,
                 latent_dim,
                 h5_filename, h5_dataset_path, sample_shape,
                 verbose=1, loss='binary_crossentropy'):

    gan = simple_gan(generator, discriminator, normal_latent_sampling((latent_dim,)))

    if(verbose>=1):
        print("="*20+" Generator "+"="*20)
        print(generator.summary())
        print("")
        print("="*20+" Discriminator "+"="*20)
        print(discriminator.summary())
        print("")
        print("="*20+" GAN "+"="*20)
        print(gan.summary())
        print("")

    model = AdversarialModel(base_model=gan,
                             player_params=[generator.trainable_weights, discriminator.trainable_weights],
                             player_names=["generator","discriminator"])

    model.adversarial_compile(adversarial_optimizer=adversarial_optimizer,
                              player_optimizers=[gen_opt, disc_opt],
                              loss=loss)

    #dbg = debugprint()

    sample_generator = util.h5_block_generator(h5_filename, h5_dataset_path, sample_shape, [1,0,0,1])

    model.fit_generator(sample_generator, 16, 5,
                        verbose=verbose,
                        validation_data=sample_generator, validation_steps=16)

    # model = Sequential()
    # model.add(Conv3D(32, (5,5,3), input_shape=(24,24,12,1), activation="relu"))
    # model.add(MaxPooling3D((2,2,2)))
    # model.add(Conv3D(16, (3,3,3), input_shape=(10,10,5,32), activation="relu"))
    # model.add(Dropout(0.4))
    # model.add(Dense(10, activation="relu"))
    # model.add(Dense(1))
    # model.add(Activation("sigmoid"))
    # print(model.summary())


def main():
    latent_dim = 300
    input_shape = (24, 24, 12)

    generator = em_generator(latent_dim, input_shape)
    discriminator = em_discriminator(input_shape)
    #print(discriminator.summary())
    #print(generator.summary())
    train_em_gan(AdversarialOptimizerSimultaneous(),
                 generator, discriminator,
                 Adam(1e-4, decay=1e-4),
                 Adam(1e-3, decay=1e-4),
                 latent_dim,
                 "/home/thistlethwaiten/cremi-data/sample_A_20160501.hdf","/volumes/raw", input_shape)


if __name__=="__main__":
    main()