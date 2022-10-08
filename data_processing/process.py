import pandas as pd 
import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
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
  if(not os.path.exists(final_path + '/train/path_pas_silos')):
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

  train_datagen = ImageDataGenerator(rescale = rescale_ratio,rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
  test_datagen = ImageDataGenerator( rescale = rescale_ratio)
  
  train = train_datagen.flow_from_directory(train_path , batch_size = _batch_size, class_mode = 'binary', target_size = (image_size, image_size))
  valid = test_datagen.flow_from_directory(validation_path,  batch_size = _batch_size, class_mode = 'binary', target_size = (image_size, image_size))

  return train,valid

def generate_test(
  test_path : str
  ):
    array_pas_silos = []
    array_silos =[]
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