"""
    Combine labels of the same media post, to form multi-label structure
    Plot number of mentions of different classes.
    Over-/under-sample data to solve class imbalance.
"""

import os
import numpy as np
import pandas as pd
from data_processing_utils import classify_products, plot_distribution, under_over_sample

BASE_DIR = "../"
DATA_DIR = "data"
MEDIA_FILE = "imbalanced_media.csv"
PRODUCTS_FILE = "imbalanced_products.csv"

media_path = os.path.join(BASE_DIR, DATA_DIR, MEDIA_FILE)
products_path = os.path.join(BASE_DIR, DATA_DIR, PRODUCTS_FILE)
media = pd.read_csv(media_path)
products = pd.read_csv(products_path)
media.fillna(value={'media_caption': 'no caption'}, inplace=True)

# Classify the productas into different classes
productID_classID, className_classID, className_productID, classID_className = classify_products(products)

# Add another column in media data frame, for fine cagegories
# labels as list, for ease of combining multi-labels of same media (later "groupby")
media.drop('class_ID', axis=1, inplace=True)
labels = []
for idx, row_data in media.iterrows():
    productID = row_data.tagged_product_id
    classID = productID_classID[productID]
    labels.append(classID)
media['label'] = labels

# Drop duplicate rows, rows with different tagged product IDs are considered different
# and will be grouped tegother later
columns_to_consider = list(set(media.columns) - set(['tagged_product_id']))
media.drop_duplicates(subset=columns_to_consider, keep='first', inplace=True)

# convert int labels to list, for ease of combining multi-labels of same media
labels = []
for idx, row_data in media.iterrows():
    label = row_data.label
    labels.append([label])
media['label'] = labels

distribution = plot_distribution(media, classID_className, multiLabels=True)

# classIDs to under-/over-sample
classIDs_to_underSample = [i[0] for i in distribution[:10]]
classIDs_to_overSample = [i[0] for i in distribution[-16:]]

# Add two additional columns before combining, to speed up re-sampling process
under_sample, over_sample = [], []
for idx, row_data in media.iterrows():
    classID = row_data.label[0]
    under = 1 if classID in classIDs_to_underSample else 0
    under_sample.append(under)
    over = 1 if (classID in classIDs_to_overSample and under == 0) else 0
    over_sample.append(over)

media['under_sample'] = under_sample
media['over_sample'] = over_sample

# Combine meida dataframe based on media_id for multi-class, multi-labels training
aggregation_functions = {column: 'first' for column in media.columns}
aggregation_functions['label'] = 'sum'
aggregation_functions['under_sample'] = 'max'
aggregation_functions['over_sample'] = 'max'

media = media.groupby(media['media_id'], as_index=False).aggregate(aggregation_functions)

for idx, row_data in media.iterrows():
    labels = row_data.label
    old_len = len(labels)
    new_len = len(list(set(labels)))
    if old_len != new_len:
        print('warning')

media_sampled = under_over_sample(media)
# need to combine "none" and "shopping_related" into "misc" for caption_type
media_sampled.loc[media_sampled.media_caption == '', 'caption_type'] = 'missing'
media_sampled.loc[media_sampled.caption_type == 'shopping_related', 'caption_type'] = 'misc'
media_sampled.loc[media_sampled.caption_type == 'none', 'caption_type'] = 'misc'

# delete the instance with short code'BpAtcr-AqEU', because image is damaged
media_sampled.drop(media_sampled[media_sampled.media_shortcode == 'BpAtcr-AqEU'].index,
                   inplace=True)
#media_sampled.to_csv(BASE_DIR + "/data/balanced_data.csv", index=False)
media_sampled.to_csv("balanced_data.csv", index=False)
