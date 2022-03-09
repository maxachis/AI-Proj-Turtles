import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

train = pd.read_csv('../train.csv')
test = pd.read_csv('../test.csv')
sample_submission = pd.read_csv('../sample_submission.csv')
src_dir = os.path.join(os.getcwd(), "..\\image_datasets", 'turtle_big_classes')
print(src_dir)




batch_size = 32
img_height = 224
img_width = 224

train_ds = tf.keras.utils.image_dataset_from_directory(
  src_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  color_mode="grayscale",
#   image_size=(img_height, img_width), #Resize
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  src_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  color_mode="grayscale",
#   image_size=(img_height, img_width), #Resize
  batch_size=batch_size)

#Standardize RGB values
normalization_layer = tf.keras.layers.Rescaling(1./255)

#Configure dataset for performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 100
IMG_SIZE = 224

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Resizing(IMG_SIZE, IMG_SIZE),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.RandomRotation(0.2),
#   tf.keras.layers.RandomCrop(int(IMG_SIZE*0.95),int(IMG_SIZE*0.95)),
#   tf.keras.layers.RandomContrast(factor=0.2),
#   tf.keras.layers.random_brightness(x, 0.2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

def run_model_and_log(model, dataset, validation_data, num_epochs):
  epoch_log = {
    "accuracy": [],
    "val_loss": []
  }
  log_acc = tf.keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epochs, logs: epoch_log["accuracy"].append(logs.get('accuracy')))
  log_loss = tf.keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epochs, logs: epoch_log["val_loss"].append(logs.get('val_loss')))
  model.fit(
    dataset,
    validation_data=validation_data,
    epochs=num_epochs,
    callbacks=[log_acc, log_loss]
  )
  return {
    "final_train_accuracy": epoch_log["accuracy"][len(epoch_log["accuracy"]) - 1],
    "best_train_accuracy": max(epoch_log["accuracy"]),
    "final_test_accuracy": -1,
    "best_test_accuracy": -1
  }

run_model_and_log(model, train_ds, val_ds, 5)