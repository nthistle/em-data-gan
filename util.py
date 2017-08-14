import h5py
import numpy as np

#credit for lheinric for most of this code
def h5_block_generator(filename, path, sample_size, batch_size=16, apply=None):
    ds = h5py.File(filename, "r")[path]

    while True:
        batch = np.empty((batch_size,) + sample_size)

        z_start = np.random.random_integers(0, ds.shape[0] - sample_size[0] - 1, batch_size)
        y_start = np.random.random_integers(0, ds.shape[1] - sample_size[1] - 1, batch_size)
        x_start = np.random.random_integers(0, ds.shape[2] - sample_size[2] - 1, batch_size)

        for k in range(batch_size):
            ds.read_direct(batch,
                           np.s_[z_start[k]:z_start[k] + sample_size[0],
                           y_start[k]:y_start[k] + sample_size[1],
                           x_start[k]:x_start[k] + sample_size[2]],
                           np.s_[k, :, :, :])

        gt = np.zeros((batch_size, 1, 1, 1))

        yield ([batch], [gt] * batch_size)


