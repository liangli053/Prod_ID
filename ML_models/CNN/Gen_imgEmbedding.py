""" Transfer learning. To read from the "balanced_data.csv" file
    and pre-compute VGG19 output and strore as 4096-dim numpy arrays.

    Pre-trained VGG 19 is used for multi-label image classification.
    The output of FC2 layer of VGG19 (right before softmax) is used.
    For faster training, the FC2 output is pre-computed and stored.
"""
import os
import numpy as np
import pandas as pd
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg19 import preprocess_input

BASE_DIR = "../.."
DATA_DIR = "data"
IMG_DIR = "images"
DATA_FILE = "balanced_data.csv"

data_path = os.path.join(BASE_DIR, DATA_DIR, DATA_FILE)
img_folder = os.path.join(BASE_DIR, IMG_DIR)
data = pd.read_csv(data_path)

# 'media_shortcode' and 'label' are the only columns that matter in image classification
data = pd.read_csv(data_path).loc[:, ['media_shortcode', 'label']]

dim=(224,224,3)

# load pre-trained VGG19 model
base_model = VGG19(weights='imagenet') #imports the mobilenet model
for layer in base_model.layers:
    layer.trainable=False

output = base_model.get_layer('fc2').output

model=Model(inputs=base_model.input, outputs=output)

code_list = list(set(data['media_shortcode']))
total = len(code_list)
for idx,  short_code in enumerate(code_list):
    if (idx+1) % 1000 == 0:
        print("%d of %d images were embedded" %(idx+1, total))
    try:
        img_path = img_folder + '/' + short_code + '.jpg'
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        CNN_output = model.predict(image)
        out_path = "../VGG19_FC2_embed/" + short_code + '.npy'
        np.save(out_path, CNN_output)
    except:
        print("File is inaccessible:", short_code)
