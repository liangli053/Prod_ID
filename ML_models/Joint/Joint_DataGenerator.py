import numpy as np
import math
import keras


class DataGenerator(keras.utils.Sequence):
    """ Two many image files, can't fit in memory
        Generates data for mini-batch training
    """

    def __init__(self, X, Y, imgID_shortCode, imgEmbd_dir, batch_size=128, caption_words=40):
        """ Initialization
            Args:
                X: training instances
                Y: labels
                imgID_shortCode: dict{int : str}, image ID -> image short code,
                                used to locate the image npy files
                imgEmbd_dir: str, folder that stores image npy files
                batch_size: int
                caption_words: number of words in caption sequence
        """
        # For the joint model, last column of X is image indices, row of Y is 40-class labels
        self.inputs = X
        self.labels = Y
        self.imgID_shortCode = imgID_shortCode  # image path identifier
        self.imgEmbd_dir = imgEmbd_dir
        self.batch_size = batch_size
        self.caption_words = caption_words

    def __len__(self):
        'The number of batches per epoch'
        return int(math.floor(len(self.inputs) / self.batch_size))

    def __getitem__(self, idx):
        """ Inheritied from kera.utils.Sequence.
            The method __getitem__ should return a complete batch.
        """
        # indices of current mini-batch
        batch_indices = list(range(idx * self.batch_size, (idx + 1) * self.batch_size))
        batch_inputs = self.inputs[batch_indices]  # batch_size x 59, last column is image indices
        batch_labels = self.labels[batch_indices]  # batch_size x 49

        # Caption has 40 words, ins_username embedded into 20 dims, categoriral 5+6+6
        caption_embedding = batch_inputs[:, :self.caption_words]
        userName_embedding = batch_inputs[:, self.caption_words]
        other_categorical = batch_inputs[:, self.caption_words + 1:-1]
        batch_imgIDs = batch_inputs[:, -1]
        batch_image_input = self.__data_generation(batch_imgIDs)

        return [caption_embedding, userName_embedding, other_categorical, batch_image_input], batch_labels

    def __data_generation(self, batch_imgIDs):
        """ Generates data containing batch_size samples
            Args:
                batch_imageIDs: numpy vector, image IDs in current batch
            Returns:
                X: numpy matrix, dim: batch_size x 4096, embedded image vectors of curret batch
        """
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
