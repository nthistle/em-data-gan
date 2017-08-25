import numpy as np
import pandas as pd
import os
import sys

from keras_adversarial import AdversarialModel, simple_gan
from keras_adversarial import normal_latent_sampling, AdversarialOptimizerSimultaneous
from keras.optimizers import Adam

import util
from em_gan_models import *


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
    gen_saver = util.SaveModel(generator, os.path.join(output_directory, "generator"))
    disc_saver = util.SaveModel(discriminator, os.path.join(output_directory, "discriminator"))

    history = model.fit_generator(sample_generator, per_epoch, epochs=epochs,
                        verbose=verbose, callbacks=[sampler, gen_saver, disc_saver],
                        validation_data=sample_generator, validation_steps=(per_epoch//5))

    df = pd.DataFrame(history.history)
    df.to_csv(os.path.join(output_directory, "history.csv"))

    discriminator.save(os.path.join(output_directory, "gan_disc_" + str(epochs) + "_" + str(per_epoch) + "_" + r_id + ".h5"))
    generator.save(os.path.join(output_directory, "gan_gen_" + str(epochs) + "_" + str(per_epoch) + "_" + r_id + ".h5"))

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
                     r_id=("large_" + str(gen_lr) + "_" + str(disc_lr) + "_" + str(gen_reg) + "_" + str(disc_reg)),
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