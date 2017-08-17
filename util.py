import h5py
import numpy as np
import os
from keras.callbacks import Callback
from scipy.misc import imresize
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


class SampleEM(Callback):

    def __init__(self, image_path, generator):
        super().__init__()
        self.image_path = image_path
        self.generator = generator

    def on_epoch_end(self, epoch, logs=None):
        if not os.path.isdir(self.image_path):
            os.mkdir(self.image_path)

        dat = self.generator() #assume dimensions of (size,24,24,12)

        print("="*15 + " EPOCH %d "%epoch + "="*15)
        for k in logs:
            print(k,logs[k])
        print("")

        format_blocks_nicely(dat).save(self.image_path + ("/epoch_%03d.png"%epoch))
        #target_dir = self.image_path + ("/epoch_%02d"%epoch)
        #if not os.path.isdir(target_dir):
        #    os.mkdir(target_dir)
        #for i in range(len(dat)):
        #    img = Image.fromarray((255*dat[i][:,:,0,0]).astype(np.uint8))
        #    img.save(target_dir + "/%02d_slice0.png"%i)
        #    img = Image.fromarray((255*dat[i][:,:,6,0]).astype(np.uint8))
        #    img.save(target_dir + "/%02d_slice6.png"%i)


#credit for lheinric for most of this code
def h5_block_generator(filename, path, sample_shape, expected_output, batch_size=16):
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