import numpy as np
import sys
from keras.layers import Reshape, Flatten, Activation
from keras.layers.core import Dense, Dropout
from keras.layers.convolutional import Conv3D, Conv3DTranspose, UpSampling3D
from keras.models import Sequential
from keras_adversarial import AdversarialModel, simple_gan
from keras_adversarial import normal_latent_sampling, AdversarialOptimizerSimultaneous
import util
from keras.optimizers import Adam


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
                 output_directory,
                 verbose=1, loss='mean_squared_error',
                 epochs=10, per_epoch=100, id="em-gan"):

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

    zsamples = np.random.normal(size=(5, latent_dim))

    sample_generator = util.h5_block_generator(h5_filename, h5_dataset_path, sample_shape, [1,0,0,1])

    def generator_sampler():
        return generator.predict(zsamples)

    sampler = util.SampleEM(output_directory,generator_sampler)

    model.fit_generator(sample_generator, per_epoch, epochs=epochs,
                        verbose=verbose, callbacks=[sampler],
                        validation_data=sample_generator, validation_steps=(per_epoch//5))

    discriminator.save("gan_disc_" + str(epochs) + "_" + str(per_epoch) + "_" + id + ".h5")
    generator.save("gan_gen_" + str(epochs) + "_" + str(per_epoch) + "_" + id +".h5")


def main(file_source, epochs, per_epoch, verbose, output_directory, loss, gen_lr, disc_lr):
    latent_dim = 300
    input_shape = (24, 24, 12)

    generator = em_generator(latent_dim, input_shape)
    discriminator = em_discriminator(input_shape)

    train_em_gan(AdversarialOptimizerSimultaneous(),
                 generator, discriminator,
                 Adam(gen_lr),
                 Adam(disc_lr),
                 latent_dim,
                 file_source,"/volumes/raw", input_shape,
                 output_directory,
                 verbose=verbose, epochs=epochs, per_epoch=per_epoch, loss=loss,
                 id=(str(gen_lr) + "_" + str(disc_lr)))


if __name__=="__main__":
    if(len(sys.argv)<7):
        print("Usage: python run_model.py [input_em_file] [epochs] [per_epoch] [verbose] [output_directory] [loss] [gen_lr] [disc_lr]")
        print("(gen_lr and disc_lr are optional)")
    else:
        main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6],
             float(sys.argv[7]) if len(sys.argv)>7 else 1e-4,
             float(sys.argv[8]) if len(sys.argv)>8 else 1e-3)