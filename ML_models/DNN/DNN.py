""" A DNN with word embedding (GloVe) for beauty productcs
    classification using textual and categorical data.
    GloVe 100-dim word embedding is used for warm start.
    Instagram username is converted to 20-dim vector via embedding.
"""
import os
import sys
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, concatenate, Flatten
from keras.layers.core import Dense, Reshape, Lambda
from keras.utils import to_categorical
from keras import regularizers
import keras.backend as K
import tensorflow as tf
from tensorflow.keras.initializers import Constant
from tensorflow import set_random_seed
import pickle
from sklearn.metrics import classification_report
sys.path.append('../')
from train_utils import OrdinalEncoder, multi_hot_encoding, split_data,\
    precision, recall, exact_match_ratio, get_precision, get_recall, get_exact_match_ratio

BASE_DIR = "../.."
DATA_DIR = "data"
DATA_FILE = "balanced_data.csv"
WORD_EMBEDDING_DIM = 100
GLOVE_DIR = "glove.6B"
GLOVE_FILE = "glove.6B." + str(WORD_EMBEDDING_DIM) + 'd.txt'
GLOVE_VOCSIZE = 400000
INS_USERNAME_EMBEDDING_DIM = 20
MAX_WORDS_IN_CAPTION = 40
SPLIT_RATIO = [0.7, 0.15, 0.15]  # training : validation : test

data_path = os.path.join(BASE_DIR, DATA_DIR, DATA_FILE)
data = pd.read_csv(data_path)

# number of ins usernames, poster account types, media types and caption types
N_ins_username = data.poster_instagram_username.nunique()
N_poster_account_type = data.poster_account_type.nunique()
N_media_type = data.media_type.nunique()
N_caption_type = data.caption_type.nunique()

# dimension of raw input for each instance
raw_input_dim = MAX_WORDS_IN_CAPTION + 1 + N_poster_account_type + N_media_type + N_caption_type

# Transform the tabular data to usable numerial data to feed in the network
# build word : vec dictionary from GloVe
glove_path = os.path.join(BASE_DIR, GLOVE_DIR, GLOVE_FILE)
glove_index = {}
with open(glove_path) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, dtype='float', sep=' ')
        glove_index[word] = coefs

# prepare tokenizer
captions = data.media_caption.astype(str)
token = Tokenizer()
token.fit_on_texts(captions)
vocab_size = len(token.word_index) + 1  # word_index is 1-indexed, puls 1-dim for OOV and padded words
print("The size of the vocabulary is: ", vocab_size)

# integer encode the documents
encoded_captions = token.texts_to_sequences(captions)
# pad documents to a max length of 4 words
padded_captions = pad_sequences(encoded_captions, maxlen=MAX_WORDS_IN_CAPTION, padding='post')

# use "represented_data" matrix, to store the encoded data
represented_data = padded_captions.copy()

# create a weight matrix for all words in captions
embedding_matrix = np.zeros((vocab_size, WORD_EMBEDDING_DIM))
for word, idx in token.word_index.items():  # word_index is 1-indexed, zero vector for padding
    embedding_vector = glove_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[idx] = embedding_vector

# encode ins usernames
ins_usernames = data.poster_instagram_username
encoded_ins_usernames = OrdinalEncoder(ins_usernames)
encoded_ins_usernames = np.array(encoded_ins_usernames).reshape(-1, 1)
# concatenate encoded_ins_usernames to represented_data
represented_data = np.concatenate((represented_data, encoded_ins_usernames), axis=1)

# Transform poster_account_type, media_type and caption_type -- one-hot
for ctgr in ['poster_account_type', 'media_type', 'caption_type']:
    col = data[ctgr]
    one_hot_encoded = to_categorical(OrdinalEncoder(col))
    represented_data = np.concatenate((represented_data, one_hot_encoded), axis=1)

# Check dimension
if len(represented_data[0]) != raw_input_dim:
    print("Input dimension is wrong!")

labels_str = data.label.values
labels = []
for str_label in labels_str:
    label = [int(i) for i in str_label[1:-1].split(',')]
    labels.append(label)

num_of_classes, encoded_labels_arr = multi_hot_encoding(labels)
represented_data = np.concatenate((represented_data, encoded_labels_arr), axis=1)

# split all data into training, validation and test sets
data_after_split = split_data(represented_data, SPLIT_RATIO, num_of_classes)

# Now the tabular data are ready to be fed into embedding layers and feed-forward network
# To ensure reproducible results
np.random.seed(42)
random.seed(42)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
tf.set_random_seed(42)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# Integer IDs representing 1-hot encodings
media_caption_in = Input(shape=(MAX_WORDS_IN_CAPTION,))
poster_instagram_username_in = Input(shape=(1,))
others_in = Input(shape=(N_poster_account_type + N_media_type + N_caption_type, ))

# Two Embedding layers
caption_embedding = Embedding(input_dim=vocab_size, output_dim=WORD_EMBEDDING_DIM,
                              embeddings_initializer=Constant(embedding_matrix),
                              input_length=MAX_WORDS_IN_CAPTION, trainable=True,
                              name='word_emb')(media_caption_in)

ins_username_embedding = Embedding(N_ins_username, INS_USERNAME_EMBEDDING_DIM,
                                   name='insID_emb')(poster_instagram_username_in)

# Reshape and merge all embeddings together
avg_caption_embedding = Lambda(lambda x: K.mean(x, axis=1, keepdims=False))(caption_embedding)

reshape_caption_embedding = Reshape(target_shape=(WORD_EMBEDDING_DIM,))
reshape_ins_username = Reshape(target_shape=(INS_USERNAME_EMBEDDING_DIM,))

combined = concatenate([reshape_caption_embedding(avg_caption_embedding),
                        reshape_ins_username(ins_username_embedding), others_in])


# Fully connected layers
regularizer_param = 0.0003
hidden_1 = Dense(64, activation='relu',
                 kernel_regularizer=regularizers.l2(regularizer_param))(combined)
hidden_2 = Dense(32, activation='relu',
                 kernel_regularizer=regularizers.l2(regularizer_param))(hidden_1)
output = Dense(num_of_classes, activation='sigmoid',
               kernel_regularizer=regularizers.l2(regularizer_param))(hidden_2)

# Compile with categorical crossentropy and adam
model = Model(inputs=[media_caption_in, poster_instagram_username_in, others_in], outputs=[output])

# Important: In multi-label classification problems. It is easy to obtain high accuracy
# because of sparsity of labels. Use customized metrics: precision, recall, and Exact Match Ratio(Subset accuracy).
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=[precision, recall, exact_match_ratio])

print(model.summary())

val_data = ([data_after_split['X_val'][:, :MAX_WORDS_IN_CAPTION],
             data_after_split['X_val'][:, MAX_WORDS_IN_CAPTION],
             data_after_split['X_val'][:, MAX_WORDS_IN_CAPTION + 1:]],
            data_after_split['Y_val'])
batchSize = 128

history = model.fit([data_after_split['X_train'][:, :MAX_WORDS_IN_CAPTION],
                     data_after_split['X_train'][:, MAX_WORDS_IN_CAPTION],
                     data_after_split['X_train'][:, MAX_WORDS_IN_CAPTION + 1:]],
                    data_after_split['Y_train'],
                    validation_data=val_data, batch_size=batchSize, epochs=50)

# save model and history
with open('DNN_history', 'wb') as fin:
    pickle.dump(history.history, fin)
model.save('DNN_model.h5')


y_pred = model.predict([data_after_split['X_test'][:, :MAX_WORDS_IN_CAPTION],
                        data_after_split['X_test'][:, MAX_WORDS_IN_CAPTION],
                        data_after_split['X_test'][:, MAX_WORDS_IN_CAPTION + 1:]])
y_true = data_after_split['Y_test']

precision, recall, = get_precision(y_true, y_pred), get_recall(y_true, y_pred)
exact_match_ratio = get_exact_match_ratio(y_true, y_pred)

print("precision, recall and exact match ratio for test set are %.4f, %.4f and %.4f" % (precision, recall, exact_match_ratio))

# print(classification_report(y_true, y_pred))
