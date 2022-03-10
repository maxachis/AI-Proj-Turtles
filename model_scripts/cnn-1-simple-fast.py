
# FIRST CNN attempt
# reaches 1.0 accuracy, but only about a max of 6.5% validation accuracy

import os

from keras.preprocessing.image import load_img, img_to_array, array_to_img
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

# Read in csv files.
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

turtle_ids = sorted(np.unique(train.turtle_id)) + ['new_turtle']
turtle_num_lookup = dict(zip(turtle_ids, np.arange(len(turtle_ids))))
turtle_id_lookup = {v: k for k, v in turtle_num_lookup.items()}
train["turtle_num"] = train["turtle_id"].map(turtle_num_lookup)

# Convert image_location strings to lowercase and ensure only allowed strings are present
# for df in [train, test]:
#   df.image_location = df.image_location.apply(lambda x: x.lower())
#   assert set(df.image_location.unique()) == set(['left', 'right', 'top'])


turtle_imgs_dir = "image_datasets/turtle_edge"
train_image_paths = [os.path.join(turtle_imgs_dir, "%s.JPG" % f) for f in train.image_id]
train_images = np.array([img_to_array(load_img(f, grayscale=True)) for f in train_image_paths]) / 255

tf.keras.backend.clear_session()

filtersize = (3,3)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, filtersize, activation='relu', input_shape=(224,224,1)),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(64, filtersize, activation='relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(64, filtersize, activation='relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(64, filtersize, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(train.turtle_id.nunique() + 1)
])

model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

model.summary()

fitstats = model.fit(train_images, train.turtle_num, epochs=10, validation_split=0.30)


