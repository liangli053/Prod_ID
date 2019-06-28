import numpy as np
import math
import keras


class DataGenerator(keras.utils.Sequence):
    """ Generates data for mini-batch training
        Two many image files, can't fit in memory
    """

    def __init__(self, X, Y, imgID_shortCode, imgEmbd_dir, batch_size=128):
        'Initialization'
        # For image-only model, row of X is image ID, row of Y is 40-class labels
        self.imageIDs = X
        self.labels = Y
        self.imgID_shortCode = imgID_shortCode  # image path identifier
        self.imgEmbd_dir = imgEmbd_dir
        self.batch_size = batch_size

    def __len__(self):
        'The number of batches per epoch'
        return int(math.floor(len(self.imageIDs) / self.batch_size))

    def __getitem__(self, idx):
        """ Inheritied from kera.utils.Sequence.
            The method __getitem__ should return a complete batch.
        """
        batch_indices = list(range(idx * self.batch_size, (idx + 1) * self.batch_size))
        batch_imgIDs = self.imageIDs[batch_indices].flatten()
        batch_labels = self.labels[batch_indices]

        # Generate data
        batch_X = self.__data_generation(batch_imgIDs)
        return batch_X, batch_labels

    def __data_generation(self, batch_imgIDs):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, 4096))
        # Generate image data
        for i, imgID in enumerate(batch_imgIDs):
            # Store sample
            short_code = self.imgID_shortCode[imgID]
            path = self.imgEmbd_dir + '/' + short_code + '.npy'
            embedded_img = np.load(path)
            X[i, :] = embedded_img
        return X
