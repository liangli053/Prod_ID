import numpy as np
from collections import Counter
import keras.backend as K
import tensorflow as tf


def OrdinalEncoder(categories):
    """ An ordinal Encoder for data pre-processing,
        Not strictly ordinal, but does the work.

        Args:
            categories: numpy arrary or list of categories
        Returns:
            numpy array of ordinal encoded entries
    """
    counter = Counter(categories).most_common()  # indices assigned based on number of occurances
    unique_ctgrs = [x[0] for x in counter]  # list of categories, guaranteed in fixed order due to sorting
    encoder = {ctgr: idx for idx, ctgr in enumerate(unique_ctgrs)}

    res = [0] * len(categories)
    res = [encoder[ctgr] for ctgr in categories]
    return np.array(res, dtype='int')


def multi_hot_encoding(labels):
    """ multi-hot encoding for the labels of each meida post
        Args:
            labels: list, dimension of rows may not be same
        Return:
            num_of_classes: int, total number of classes
            res: array, num_of_examples x num_of_classes, multi-hot encoded array
    """
    num_examples, min_idx, max_idx = len(labels), float('inf'), -float('inf')
    for row in labels:
        for col in row:
            if col < min_idx:
                min_idx = col
            if col > max_idx:
                max_idx = col
    num_of_classes = max_idx - min_idx + 1

    res = np.zeros((num_examples, max_idx - min_idx + 1), dtype=int)
    for row in range(len(labels)):
        lbls = labels[row]
        for col in lbls:
            res[row, col] = 1
    return num_of_classes, res


def split_data(orig_data, split_ratio, num_of_classes):
    """ Split dataframe into train, validation and test datasets.

        Args:
            orig_data: original pandas df
            split_ratio: list(float), training - validation - test ratio
            num_of_classes: int, number of classes
        Returns:
            data_after_split: dict, {X_train: 2D array, Y_train: 2D array,
                                     X_val: 2D array, Y_val: 2D array,
                                     X_test: 2D array, Y_test: 2D array,}
    """
    np.random.seed(42)
    np.random.shuffle(orig_data)
    data_after_split, tmp, length = {}, {}, len(orig_data)

    tmp['train'], tmp['val'], tmp['test'] = np.split(
        orig_data, [int(split_ratio[0] * length), int(1 - split_ratio[2] * length)], axis=0)

    for sett in ['train', 'val', 'test']:
        X = tmp[sett][:, :-num_of_classes]
        Y = tmp[sett][:, -num_of_classes:]
        data_after_split['X' + '_' + sett] = X
        data_after_split['Y' + '_' + sett] = Y

    print(data_after_split['X_train'].shape, data_after_split['Y_train'].shape,
          data_after_split['X_val'].shape, data_after_split['Y_val'].shape,
          data_after_split['X_test'].shape, data_after_split['Y_test'].shape)
    return data_after_split


def precision(y_true, y_pred):
    """ Precision metric.
        Args:
            y_true: tf tensor, ground-truth labels
            y_pred: tf tensor, predicted labels
        Returns:
            precision: tf tensor
    """
    y_pred = K.cast(K.greater(y_pred, 0.5), dtype=float)
    true_positives = K.sum(K.round(y_true * y_pred))
    predicted_positives = K.sum(K.round(y_pred))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """ Recall metric.
        Args:
            y_true: tf tensor, ground-truth labels
            y_pred: tf tensor, predicted labels
        Returns:
            recall: tf tensor
    """
    y_pred = K.cast(K.greater(y_pred, 0.5), dtype=float)
    true_positives = K.sum(K.round(y_true * y_pred))
    actural_positives = K.sum(K.round(y_true))
    recall = true_positives / (actural_positives + K.epsilon())
    return recall

# def Hamming_loss(y_true, y_pred):
#    # y_pred is not filterd by 0.5 yet
#    tmp = K.abs(y_true-y_pred)
#    return K.mean(K.cast(K.greater(tmp,0.5),dtype=float))


def exact_match_ratio(y_true, y_pred):
    """ Exact match ratio metric.
        Args:
            y_true: tf tensor, ground-truth labels
            y_pred: tf tensor, predicted labels
        Returns:
            exact match ratio, tf tensor
    """
    predictions = tf.to_float(tf.greater_equal(y_pred, 0.5))
    pred_match = tf.equal(predictions, tf.round(y_true))
    exact_match = tf.reduce_min(tf.to_float(pred_match), axis=1)
    return tf.reduce_mean(exact_match)


def get_precision(y_true, y_pred):
    """ Calculate precision of test set

        Args:
            y_true: 2D np array, ground-truth labels
            y_pred: 2D np array, predicted labels
        Returns:
            precision: float
    """
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    true_positives = np.sum(y_pred * y_true)
    predicted_positives = np.sum(y_pred)
    return true_positives / predicted_positives


def get_recall(y_true, y_pred):
    """ Calculate recall of test set

        Args:
            y_true: 2D np array, ground-truth labels
            y_pred: 2D np array, predicted labels
        Returns:
            recall: float
    """
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    true_positives = np.sum(y_pred * y_true)
    actural_positives = np.sum(y_true)
    return true_positives / actural_positives


def get_exact_match_ratio(y_true, y_pred):
    """ Calculate exact match ratio of test set

        Args:
            y_true: 2D np array, ground-truth labels
            y_pred: 2D np array, predicted labels
        Returns:
            exact match ratio: float
    """
    match = np.equal(y_pred, y_true)
    match = np.amin(match, axis=1)
    exact_match_ratio = np.mean(match)
    return exact_match_ratio
