import h5py
import numpy as np
import os
from keras.callbacks import Callback
from PIL import Image

class SampleEM(Callback):

    def __init__(self, image_path, generator):
        super().__init__()
        self.image_path = image_path
        self.generator = generator

    def on_epoch_begin(self, epoch, logs=None):
        if(epoch==0):
            print("(begin)")
            self.on_epoch_end(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        if not os.path.isdir(self.image_path):
            os.mkdir(self.image_path)

        dat = self.generator() #assume dimensions of (size,24,24,12)

        print("="*15 + " EPOCH %d "%epoch + "="*15)
        for k in logs:
            print(k,logs[k])
        print("")

        target_dir = self.image_path + ("/epoch_%02d"%epoch)
        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)
        for i in range(len(dat)):
            img = Image.fromarray((255*dat[i][:,:,0,0]).astype(np.uint8))
            img.save(target_dir + "/%02d_slice0.png"%i)
            img = Image.fromarray((255*dat[i][:,:,6,0]).astype(np.uint8))
            img.save(target_dir + "/%02d_slice6.png"%i)


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