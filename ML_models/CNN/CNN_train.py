import os
import random
import sys
import numpy as np
import pandas as pd
from keras import regularizers
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense
from keras.applications.vgg19 import VGG19
import keras.backend as K
import tensorflow as tf
from tensorflow import set_random_seed
import matplotlib.pyplot as plt
from CNNDataGenerator import DataGenerator
import math
import pickle
from sklearn.metrics import classification_report
sys.path.append('../')
from train_utils import multi_hot_encoding, split_data, precision, \
    recall, exact_match_ratio, get_precision, get_recall, get_exact_match_ratio

BASE_DIR = "../.."
DATA_DIR = "data"
IMG_DIR = "images"
IMG_EMBEDDED_DIR = "VGG19_FC2_embed"
DATA_FILE = "balanced_data.csv"
SPLIT_RATIO = [0.7, 0.15, 0.15]  # training : validation : test

data_path = os.path.join(BASE_DIR, DATA_DIR, DATA_FILE)
img_path = os.path.join(BASE_DIR, IMG_DIR)
imgEmbd_dir = os.path.join(BASE_DIR, 'src', IMG_EMBEDDED_DIR)

# 'media_shortcode' and 'label' are the only columns that matter in image classification
data = pd.read_csv(data_path).loc[:, ['media_shortcode', 'label']]
data.reset_index(drop=False, inplace=True)
# build a imgID-img_shortCode dictionary, since data will be stored in a numpy array
imgID_shortCode = {idx: code for idx, code in zip(data.index, data.media_shortcode)}

# Multi-hot encoding for multi-labels
represented_data = np.array(data.index).reshape(-1, 1)
labels_str = data.label.values
labels = []
for str_label in labels_str:
  label = [int(i) for i in str_label[1:-1].split(',')]
  labels.append(label)

num_of_classes, encoded_labels_arr = multi_hot_encoding(labels)
represented_data = np.concatenate((represented_data, encoded_labels_arr), axis=1)
data_after_split = split_data(represented_data, SPLIT_RATIO, num_of_classes)

##------ Train CNN -----##
# To ensure reproducible results
np.random.seed(42)
random.seed(42)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
tf.set_random_seed(42)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# Transfer learning based on VGG19, output from 'fc2' layer
# Embeddings stored at ../VGG19_embeddImgs
# Add fully connected layers

# Embedded image vector from VGG19
image_in = Input(shape=(4096,))

# Add fully connected layers
regularizer_param = 0.0001
x = Dense(1024, activation='relu',
          kernel_regularizer=regularizers.l2(regularizer_param))(image_in)
x = Dense(512, activation='relu',
          kernel_regularizer=regularizers.l2(regularizer_param))(x)
x = Dense(128, activation='relu',
          kernel_regularizer=regularizers.l2(regularizer_param))(x)

output = Dense(num_of_classes, activation='sigmoid',
               kernel_regularizer=regularizers.l2(regularizer_param))(x)  # final layer with softmax activation

model = Model(inputs=image_in, outputs=output)

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=[precision, recall, exact_match_ratio])


print(model.summary())

batch_size = 128
training_generator = DataGenerator(data_after_split['X_train'], data_after_split['Y_train'],
                                   imgID_shortCode, imgEmbd_dir, batch_size)

validation_generator = DataGenerator(data_after_split['X_val'], data_after_split['Y_val'],
                                     imgID_shortCode, imgEmbd_dir, batch_size)

# validation_steps: Total number of steps (batches of samples) to yield from validation_data generator
# before stopping at the end of every epoch. It should typically be equal to the number of samples of
# your validation dataset divided by the batch size.
history = model.fit_generator(generator=training_generator, validation_data=validation_generator,
                              steps_per_epoch=int(math.ceil(len(data_after_split['X_train']) / batch_size)),
                              validation_steps=int(math.ceil(len(data_after_split['X_val']) / batch_size)),
                              shuffle=True, epochs=2)

with open('CNN_history', 'wb') as fin:
  pickle.dump(history.history, fin)
model.save('CNN_model.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

# test set data generator, note tht batch size = number of samples
test_generator = DataGenerator(data_after_split['X_test'], data_after_split['Y_test'],
                               imgID_shortCode, imgEmbd_dir, batch_size=len(data_after_split['X_test']))

y_pred = model.predict_generator(generator=test_generator)
y_true = data_after_split['Y_test']

precision, recall, = get_precision(y_true, y_pred), get_recall(y_true, y_pred)
exact_math_ratio = get_exact_match_ratio(y_true, y_pred)

print("precision, recall and exact match ratio for test set are %.4f, %.4f and %.4f" % (precision, recall, exact_math_ratio))

classID_className = {0: 'color cosmetics:eye:eyebrow', 1: 'skincare:skincare:moisturizer', 2: 'color cosmetics:eye:eyeshadow', 3: 'color cosmetics:eye:mascara', 4: 'accessories:accessories:brush', 5: 'fragrance:fragrance:fragrance', 6: 'skincare:skincare:cleanser', 7: 'accessories:accessories:tool', 8: 'nail:nail:nail polish', 9: 'color cosmetics:eye:eye palette', 10: 'bath body:bath body:wash', 11: 'hair:style:styling products', 12: 'skincare:skincare:treatments', 13: 'color cosmetics:face:powder', 14: 'skincare:skincare:mask', 15: 'bath body:bath body:body lotion', 16: 'hair:cleanse:conditioner', 17: 'color cosmetics:cheek:cheek palette', 18: 'color cosmetics:lip:lipstick', 19: 'hair:treat:hair treatments', 20: 'color cosmetics:cheek:highlighter', 21: 'hair:cleanse:shampoo', 22: 'color cosmetics:face:setting spray', 23: 'color cosmetics:cheek:blush', 24: 'skincare:skincare:face suncare', 25: 'color cosmetics:eye:eyeliner', 26: 'color cosmetics:face:face palette', 27: 'color cosmetics:face:foundation', 28: 'color cosmetics:lip:lip balm', 29: 'skincare:skincare:face mist', 30: 'skincare:skincare:eyecare', 31: 'color cosmetics:eye:lash', 32: 'color cosmetics:lip:lip gloss', 33: 'color cosmetics:face:face primer', 34: 'color cosmetics:face:concealer', 35: 'color cosmetics:cheek:bronzer', 36: 'skincare:skincare:toner', 37: 'color cosmetics:lip:lip liner', 38: 'bath body:bath body:body suncare', 39: 'bath body:bath body:body glitter'}
target_names = [i.split(':')[2] for i in classID_className.values()]
print(classification_report(y_true, y_pred, target_names=target_names))
