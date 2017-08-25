import h5py
import numpy as np
import os
from keras.callbacks import Callback
from scipy.misc import imresize
from scipy.ndimage.filters import gaussian_filter
from PIL import Image

#block should be of shape (24,24,12)
def format_block_to_slices(data_block):
    arr = np.full((26,151), 0.999)
    for i in range(6):
        t = data_block[:,:,2*i].reshape((24,24)) # gets rid of trailing dim of (1,)
        arr[1:25,1+25*i:25+25*i] = t
    return arr


#data blocks should be of shape (x,24,24,12)
def format_blocks_nicely(data_blocks):
    img = np.full((26*data_blocks.shape[0],151), 0.999)
    for i in range(data_blocks.shape[0]):
        img[26*i:26+26*i,:] = format_block_to_slices(data_blocks[i])
    return Image.fromarray(imresize((np.clip(256*img,0,255)).astype(np.uint8), (3*26*data_blocks.shape[0], 3*151)))


#block should be of shape (64,64,7)
def format_large_block_to_slices(data_block):
    arr = np.full((66,456), 0.999)
    for i in range(7):
        t = data_block[:,:,i].reshape((64,64)) # gets rid of trailing dim of (1,)
        arr[1:65,1+65*i:65+65*i] = t
    return arr


#data blocks should be of shape (x,24,24,12)
def format_large_blocks_nicely(data_blocks):
    img = np.full((66*data_blocks.shape[0],456), 0.999)
    for i in range(data_blocks.shape[0]):
        img[66*i:66+66*i,:] = format_large_block_to_slices(data_blocks[i])
    return Image.fromarray(imresize((np.clip(256*img,0,255)).astype(np.uint8), (66*data_blocks.shape[0], 456)))


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


def print_model_summaries(generator, discriminator, gan):
    print("="*20+" Generator "+"="*20)
    generator.summary()
    print("")
    print("="*20+" Discriminator "+"="*20)
    discriminator.summary()
    print("")
    print("="*20+" GAN "+"="*20)
    gan.summary()
    print("")


class SaveModel(Callback):
    # base name should include directory, and beginning of file name
    # _[epoch].h5 will be postpended before saving
    def __init__(self, model_to_save, base_name, freq=5):
        super().__init__()
        self.model_to_save = model_to_save
        self.base_name = base_name
        self.freq = freq

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1)%self.freq==0:
            self.model_to_save.save(self.base_name + "_" + str(epoch) + ".h5")


class SampleEM(Callback):

    def __init__(self, image_path, generator, is_large=False, info_print=False):
        super().__init__()
        self.image_path = image_path
        self.generator = generator
        self.is_large = is_large
        self.info_print = info_print

    def on_epoch_end(self, epoch, logs=None):
        if not os.path.isdir(self.image_path):
            os.mkdir(self.image_path)

        dat = self.generator()

        if(self.info_print):
            print("Epoch #%d "%epoch)
            for k in logs:
                print(k,logs[k])
            print()

        if self.is_large:
            format_large_blocks_nicely(dat).save(self.image_path + ("/epoch_%03d.png"%epoch))
        else:
            format_blocks_nicely(dat).save(self.image_path + ("/epoch_%03d.png"%epoch))


def h5_boundary_block_generator(data_filename, data_path, bound_filename, bound_path, sample_shape, expected_output, sigma=2.5, batch_size=16, seed=None):
    if seed:
        np.random.seed(seed)

    data_ds = h5py.File(data_filename, "r")[data_path]
    bound_ds = h5py.File(bound_filename, "r")[bound_path]

    boundaries = 255.*np.array(bound_ds)
    boundaries_smooth = gaussian_filter(boundaries, [0.1*sigma, sigma, sigma])

    while True:
        batch = np.empty((batch_size,) + (sample_shape[2], sample_shape[0], sample_shape[1]) + (2,))

        # assume dataset stores like (z,y,x) (although it might actually be zxy, doesn't matter)
        z_start = np.random.random_integers(0, data_ds.shape[0] - sample_shape[2] - 1, batch_size)
        y_start = np.random.random_integers(0, data_ds.shape[1] - sample_shape[1] - 1, batch_size)
        x_start = np.random.random_integers(0, data_ds.shape[2] - sample_shape[0] - 1, batch_size)

        for k in range(batch_size):
            data_ds.read_direct(batch,
                           np.s_[z_start[k]:z_start[k] + sample_shape[2],
                           y_start[k]:y_start[k] + sample_shape[1],
                           x_start[k]:x_start[k] + sample_shape[0]],
                           np.s_[k, :, :, :, 0])

            batch[k,:,:,:,1] = boundaries_smooth[z_start[k]:z_start[k] + sample_shape[2],
                               y_start[k]:y_start[k] + sample_shape[1],
                               x_start[k]:x_start[k] + sample_shape[0]]

        batch = np.swapaxes(batch, 1, 3)

        yield ([batch/255.], [np.ones((batch_size, 1)) if x==1 else np.zeros((batch_size, 1)) for x in expected_output])


#credit for lheinric for most of this code
def h5_block_generator(filename, path, sample_shape, expected_output, batch_size=16, seed=None):
    if seed:
        np.random.seed(seed)

    ds = h5py.File(filename, "r")[path]

    while True:
        batch = np.empty((batch_size,) + (sample_shape[2], sample_shape[0], sample_shape[1]))

        # assume dataset stores like (z,y,x) (although it might actually be zxy, doesn't matter)
        z_start = np.random.random_integers(0, ds.shape[0] - sample_shape[2] - 1, batch_size)
        y_start = np.random.random_integers(0, ds.shape[1] - sample_shape[1] - 1, batch_size)
        x_start = np.random.random_integers(0, ds.shape[2] - sample_shape[0] - 1, batch_size)

        for k in range(batch_size):
            ds.read_direct(batch,
                           np.s_[z_start[k]:z_start[k] + sample_shape[2],
                           y_start[k]:y_start[k] + sample_shape[1],
                           x_start[k]:x_start[k] + sample_shape[0]],
                           np.s_[k, :, :, :])

        batch = np.swapaxes(batch, 1, 3)

        yield ([np.reshape(batch/255., batch.shape + (1,))],
                       [np.ones((batch_size, 1)) if x==1 else np.zeros((batch_size, 1)) for x in expected_output])