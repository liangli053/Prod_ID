"""
    Read media and products csv file, clean up data
    Classify products into multiple bins and regenerate labels for media posts
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict
from data_processing_utils import classify_products, plot_distribution, keepGloVeWords, cleaner,\
    remove_emoji, cleanunderscore, normalize

BASE_DIR = ".."
DATA_DIR = "data/orig_data"
MEDIA_FILE = "20190610_tagged_media.csv"
PRODUCTS_FILE = "20190610_product_labels.csv"

# specify paths to media, products and image files
media_path = os.path.join(BASE_DIR, DATA_DIR, MEDIA_FILE)
products_path = os.path.join(BASE_DIR, DATA_DIR, PRODUCTS_FILE)
image_dir = os.path.join(BASE_DIR, 'images')

media = pd.read_csv(media_path)
products = pd.read_csv(products_path)

# Fill in the missing values
values_to_fill = {'sub_category': 'unclear', 'product_category': 'unclear', 'product_page_url': 'none',
                  'product_image_url': 'none'}
products.fillna(value=values_to_fill, inplace=True)

values_to_fill = {'media_caption': 'no caption', 'media_type': 'missing', 'caption_type': 'missing'}
media.fillna(value=values_to_fill, inplace=True)

# To reduce the number of classes, for balanced distribution
# remove products with unclear categories
categories = ["sub_category", "product_category"]
indices_to_drop = []
for ctgr in categories:
    indices_to_drop.extend(list(products[products[ctgr] == 'unclear'].index))

indices_to_drop = set(indices_to_drop)
products.drop(indices_to_drop, axis=0, inplace=True)

# Some tagged products IDs are not in product list, delete them from media data frame
productIDs_in_inventory = products.product_id.unique()
productIDs_in_media = media.tagged_product_id.unique()
not_in_inventory = set(filter(lambda x: x not in productIDs_in_inventory, productIDs_in_media))
media = media[~media.tagged_product_id.isin(not_in_inventory)]

# Some social media images are not availale, exclude these posts for now
# All downloadable images are stored locally
good_shortCodes = set([filename.split('.')[0] for filename in os.listdir(image_dir)])
media = media[media.media_shortcode.isin(good_shortCodes)]

# Delete the products that are not tagged in the media file, as they are not going to be learned anyway
productIDs_in_media = media.tagged_product_id.unique()
products = products[products.product_id.isin(productIDs_in_media)]

# Classify the productas into different classes
productID_classID, className_classID, className_productID, classID_className = classify_products(products)

# Add another column in media data frame, for fine cagegories
labels = []
for idx, row_data in media.iterrows():
    productID = row_data.tagged_product_id
    classID = productID_classID[productID]
    labels.append(classID)
media['class_ID'] = labels

# plot the distribution of number of mentions for each class
distribution = plot_distribution(media, classID_className, multiLabels=False)

classIDs_to_keep = [i[0] for i in distribution]
productIDs_to_keep = []
for classID in classIDs_to_keep:
    className = classID_className[classID]
    productIDs_to_keep.extend(className_productID[className])

# update produtcts and media data frame
products = products[products.product_id.isin(productIDs_to_keep)]
media = media[media.tagged_product_id.isin(productIDs_to_keep)]

# caption cleaning
GloVe_words = set()
with open(os.path.join(BASE_DIR, 'glove.6B/glove.6B.50d.txt')) as f:
    for line in f:
        word = line.split()[0]
        GloVe_words.add(word)

print("go to caption cleaning")

captions = media.media_caption
captions = captions.astype(str).apply(remove_emoji).apply(cleaner).apply(cleanunderscore).apply(normalize).apply(keepGloVeWords, GloVe_words=GloVe_words)
media['media_caption'] = captions
media.fillna(value={'media_caption': 'no caption'}, inplace=True)
media.to_csv(BASE_DIR + "/data/imbalanced_media.csv", index=False)
products.to_csv(BASE_DIR + "/data/imbalanced_products.csv", index=False)
