import numpy as np
import pandas as pd
import os
import sys
from keras.layers import Reshape, Flatten, Activation
from keras.layers.core import Dense, Dropout
from keras.layers.convolutional import Conv3D, Conv3DTranspose, UpSampling3D
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras_adversarial import AdversarialModel, simple_gan
from keras_adversarial import normal_latent_sampling, AdversarialOptimizerSimultaneous
from keras_adversarial.legacy import l1l2
import util
from keras.optimizers import Adam


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

    #disc.add(UpSampling3D((1,1,2), input_shape=(input_shape+(1,))))

    disc.add(Conv3D(128, (7,7,2), input_shape=(input_shape+(1,)), kernel_regularizer=reg()))
    disc.add(LeakyReLU(leaky_alpha[0]))

    disc.add(Conv3D(64, (5,5,2), kernel_regularizer=reg()))
    disc.add(LeakyReLU(leaky_alpha[1]))

    disc.add(Conv3D(64, (5,5,2), kernel_regularizer=reg()))
    disc.add(LeakyReLU(leaky_alpha[2]))

    disc.add(Conv3D(64, (3,3,2), kernel_regularizer=reg()))
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



def train_em_gan(adversarial_optimizer,
                 generator, discriminator, gen_opt, disc_opt,
                 latent_dim,
                 h5_filename, h5_dataset_path, sample_shape,
                 output_directory,
                 verbose=1, loss='mean_squared_error',
                 epochs=10, per_epoch=100, r_id="em-gan",
                 is_large_model=False):

    gan = simple_gan(generator, discriminator, normal_latent_sampling((latent_dim,)))

    if verbose>=1:
        print("="*20+" Generator "+"="*20)
        generator.summary()
        print("")
        print("="*20+" Discriminator "+"="*20)
        discriminator.summary()
        print("")
        print("="*20+" GAN "+"="*20)
        gan.summary()
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

    sampler = util.SampleEM(output_directory,generator_sampler,is_large_model)
    gen_saver = util.SaveModel(discriminator, os.path.join(output_directory, "generator"))
    disc_saver = util.SaveModel(generator, os.path.join(output_directory, "generator"))

    history = model.fit_generator(sample_generator, per_epoch, epochs=epochs,
                        verbose=verbose, callbacks=[sampler, gen_saver, disc_saver],
                        validation_data=sample_generator, validation_steps=(per_epoch//5))

    df = pd.DataFrame(history.history)
    df.to_csv(os.path.join(output_directory, "history.csv"))

    discriminator.save("gan_disc_" + str(epochs) + "_" + str(per_epoch) + "_" + r_id + ".h5")
    generator.save("gan_gen_" + str(epochs) + "_" + str(per_epoch) + "_" + r_id + ".h5")

    del model
    del discriminator
    del generator


def print_model_parameters(file_source, epochs, per_epoch, verbose, output_directory, loss, gen_lr, disc_lr, gen_reg, disc_reg):
    print("Parameters:")
    print("file source =",file_source)
    print("epochs =",epochs)
    print("per_epoch =",per_epoch)
    print("verbose =",verbose)
    print("output_directory =",output_directory)
    print("loss =",loss)
    print("gen_lr =",gen_lr)
    print("disc_lr =",disc_lr)
    print("gen_reg =",gen_reg)
    print("disc_reg =",disc_reg)


def main(is_large_model, file_source, epochs, per_epoch, verbose, output_directory, loss, gen_lr, disc_lr, gen_reg, disc_reg):

    if is_large_model:
        print("Initiating large model...")
        print_model_parameters(file_source, epochs, per_epoch, verbose, output_directory, loss, gen_lr, disc_lr, gen_reg, disc_reg)
        sys.stdout.flush()

        latent_dim = 600
        input_shape = (64, 64, 7)

        generator = em_generator_large(latent_dim, input_shape, reg = lambda: l1l2(gen_reg, gen_reg))
        discriminator = em_discriminator_large(input_shape, reg = lambda: l1l2(disc_reg, disc_reg))

        train_em_gan(AdversarialOptimizerSimultaneous(),
                     generator, discriminator,
                     Adam(gen_lr),
                     Adam(disc_lr),
                     latent_dim,
                     file_source, "/volumes/raw", input_shape,
                     output_directory,
                     verbose=verbose, epochs=epochs, per_epoch=per_epoch, loss=loss,
                     r_id=("large_" + str(gen_lr) + "_" + str(disc_lr)),
                     is_large_model=True)

    else:
        print("Initiating small model...")
        print_model_parameters(file_source, epochs, per_epoch, verbose, output_directory, loss, gen_lr, disc_lr, gen_reg, disc_reg)
        sys.stdout.flush()

        latent_dim = 300
        input_shape = (24, 24, 12)

        generator = em_generator(latent_dim, input_shape, reg = lambda: l1l2(gen_reg, gen_reg))
        discriminator = em_discriminator(input_shape, reg = lambda: l1l2(disc_reg, disc_reg))

        train_em_gan(AdversarialOptimizerSimultaneous(),
                     generator, discriminator,
                     Adam(gen_lr),
                     Adam(disc_lr),
                     latent_dim,
                     file_source,"/volumes/raw", input_shape,
                     output_directory,
                     verbose=verbose, epochs=epochs, per_epoch=per_epoch, loss=loss,
                     r_id=(str(gen_lr) + "_" + str(disc_lr)))


if __name__=="__main__":
    if len(sys.argv)<8:
        print("Usage: python run_model.py [large|small] [input_em_file] [epochs] [per_epoch] [verbose] [output_directory] [loss] [gen_lr] [disc_lr] [gen_reg] [disc_reg]")
        print("(gen_lr and disc_lr are optional)")
    else:
        main(sys.argv[1].lower()[0]=="l", sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], sys.argv[7],
             float(sys.argv[8]) if len(sys.argv)>8 else 1e-4,
             float(sys.argv[9]) if len(sys.argv)>9 else 1e-3,
             float(sys.argv[10]) if len(sys.argv)>10 else 1e-6,
             float(sys.argv[11]) if len(sys.argv)>11 else 1e-6)