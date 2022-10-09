import pandas as pd 
import numpy as np
import os
import shutil
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
def create_label_folder(
  path : str,
  path_csv : str,
  final_path : str ) -> None : 
  """create folders silos and pas_silos in order to prepare the training of the model """

  df = pd.read_csv(path_csv)
  if(not os.path.exists(final_path + 'train/silos')):
    os.mkdir(final_path + 'train/')
    os.mkdir(final_path + 'validation/')
    os.mkdir(final_path + 'test/')
    os.mkdir(final_path + 'train/silos')
    os.mkdir(final_path + 'validation/silos')
    os.mkdir(final_path + 'test/silos/')
  if(not os.path.exists(final_path + '/train/pas_silos')):
    os.mkdir(final_path + 'train/pas_silos/')
    os.mkdir(final_path + 'validation/pas_silos/')
    os.mkdir(final_path + 'test/pas_silos/')
  
  for i,row in df.iterrows():
    # It is unnecessary to check if file exists since if it it exists it is squeezed
    if row[1] == 0 :
      shutil.copy(path +'/' + row.filename, f"{final_path}/{row.split}/pas_silos/")
    else:
      shutil.copy(path + '/' + row.filename,f"{final_path}/{row.split}/silos/")
  return None

def generate_train(
    train_path :str,
    validation_path :str,
    _batch_size :int = 20,
    rescale_ratio : float = 1./255.,
    image_size : int = 256
  ):

  """ Generate augmented data in order to feed more data to the models """
  train_datagen = ImageDataGenerator(rescale = rescale_ratio,rotation_range = 30, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
  test_datagen = ImageDataGenerator( rescale = rescale_ratio)
  
  train = train_datagen.flow_from_directory(train_path , batch_size = _batch_size, class_mode = 'binary', target_size = (image_size, image_size))
  valid = test_datagen.flow_from_directory(validation_path,  batch_size = _batch_size, class_mode = 'binary', target_size = (image_size, image_size))

  return train,valid

def generate_test(
  test_path : str,
  classified : bool = True,
  ):

  """ Transform test image into tensors which can be predicted by the models. 
  Returns an array of tensorflow.tensors 
  """
  array_pas_silos = []
  array_silos =[]
  array_unclassified = []
  if(classified):
    for file in os.listdir(test_path +'/silos/'):
      path_and_file = test_path +'/silos/' + file
      img = tf.io.read_file(path_and_file)
      tensor = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.float32)
      tensor = tf.expand_dims(tensor, axis=0)
      array_silos.append(tensor)
    for file in os.listdir(test_path +'/pas_silos/'):
      path_and_file = test_path +'/pas_silos/' + file
      img = tf.io.read_file(path_and_file)
      tensor = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.float32)
      tensor = tf.expand_dims(tensor, axis=0)
      array_pas_silos.append(tensor)
    return [array_silos,array_pas_silos]
  else:
    for file in os.listdir(test_path):
      path_and_file = test_path + file
      img = tf.io.read_file(path_and_file)
      tensor = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.float32)
      tensor = tf.expand_dims(tensor, axis=0)
      array_unclassified.append(tensor)
      return array_unclassified

class Silos(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, input_mask_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.input_mask_paths = input_mask_paths

    def __len__(self):
        return len(self.input_mask_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.input_mask_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size , dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size , dtype="float32")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            y[j] /= 255
        return x, y

def process_data_for_segmentation(path):

  """Generate dataset of Silos objects in order to train and test the segmentation model"""
  image_path = os.path.join(path, 'images/')
  mask_path = os.path.join(path, 'masks/')
  xai_data = pd.read_csv(os.path.join(path,'x-ai_data.csv'))
  train = [row[0] for (i,row) in xai_data.iterrows() if (row[2]=='train' and row[1]==1)]
  valid = [row[0] for (i,row) in xai_data.iterrows() if (row[2]=='validation' and row[1]==1)]
  test = [row[0] for (i,row) in xai_data.iterrows() if (row[2]=='test')]

  image_list_train = [image_path+i for i in train]
  mask_list_train = [mask_path+i for i in train]
  image_list_test = [image_path+i for i in test]
  mask_list_test = [mask_path+i for i in test]
  image_list_valid = [image_path+i for i in valid]
  mask_list_valid = [mask_path+i for i in valid]

  datasets_train = Silos(batch_size=32,img_size=(256,256,3),input_img_paths = image_list_train,input_mask_paths = mask_list_train)
  datasets_valid = Silos(batch_size=32,img_size=(256,256,3),input_img_paths = image_list_valid,input_mask_paths = mask_list_valid)
  return datasets_train, datasets_valid
    
def create_mask(pred_mask, epsilon=0.92):
  """ Generate a mask from the image given by the model"""
  max = np.max(pred_mask)
  eps = max*epsilon
    
  pred_mask = pred_mask > (max-eps)
  return pred_mask
