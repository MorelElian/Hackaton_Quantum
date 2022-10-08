import tensorflow as tf

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras.layers as layers

from tensorflow.keras.applications.vgg16 import VGG16


def generate_inception(train_generator, validation_generator):
  base_model = InceptionV3(input_shape = (256, 256, 3), include_top = False, weights = 'imagenet')

  for layer in base_model.layers:
    layer.trainable = False

  x = layers.Flatten()(base_model.output)
  x = layers.Dense(1024, activation='relu')(x)
  x = layers.Dropout(0.2)(x)

  # Add a final sigmoid layer with 1 node for classification output
  x = layers.Dense(1, activation='sigmoid')(x)

  model = tf.keras.models.Model(base_model.input, x)

  model.compile(optimizer = RMSprop(lr=0.0001), loss = 'binary_crossentropy', metrics = ['acc'])

  nc_history = model.fit_generator(train_generator, validation_data = validation_generator, steps_per_epoch = 50, epochs = 5)

  return model



def generate_VGG(train_generator, validation_generator):
  base_model = VGG16(input_shape = (256, 256, 3), # Shape of our images
  include_top = False, # Leave out the last fully connected layer
  weights = 'imagenet')

  for layer in base_model.layers:
    layer.trainable = False

  # Flatten the output layer to 1 dimension
  x = layers.Flatten()(base_model.output)

  # Add a fully connected layer with 512 hidden units and ReLU activation
  x = layers.Dense(512, activation='relu')(x)

  # Add a dropout rate of 0.5
  x = layers.Dropout(0.5)(x)

  # Add a final sigmoid layer with 1 node for classification output
  x = layers.Dense(1, activation='sigmoid')(x)

  model = tf.keras.models.Model(base_model.input, x)

  model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.0001), loss = 'binary_crossentropy',metrics = ['acc'])