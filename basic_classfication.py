import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

print('shape of train images', train_images.shape)
print('shape of test images', test_images.shape)

print('length of train labels', len(train_labels))
print('length of test labels', len(test_labels))

train_images = train_images / 255.0
test_images = test_images / 255.0

#Build the model
model = keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(train_images[2],train_images[2])),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)
])
#compile the model
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#train the model
model.fit(train_images, train_labels, epochs=5)

#evaluate accuracy
test_loss, test_acc = model.evaluate(test_image, test_labels)

print('Test accuracy:', test_acc)

predictions = model.predict(test_images)

print('prediction', predictions[0])
print('test_labels', test_labels[0])
