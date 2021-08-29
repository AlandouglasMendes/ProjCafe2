from __future__ import print_function
# Importing Image class from PIL module, nd module PIL
from PIL import Image
import PIL

import keras
#importing lib numpy
import numpy as np

#os allow us to manipulate dir, folders, files
import os

#Importing tensorflow for building/training models
import tensorflow as tf

#Importing tensrflow datasets
import tensorflow_datasets as tfds

#lib for dir manipulating
import pathlib

#image manipulation
from imageio import imread, imwrite



#dir is the path to a diretory where images will be processed, the result being output in the same dir
def procesImage(dir):
    directory = r'/' + dir + '/'
    dirPath = dir
    for filename in os.listdir(directory):
        imName = dirPath + filename
        imNonProcesssed = Image.open(imName)  
        width, height = imNonProcesssed.size
        left = (width/ 2.694) + 200
        top = (height / 5.694) + 300
        right = (width / 2.694) + 908
        bottom = (height / 5.694) + 1008  
        im1 = imNonProcesssed.crop((left, top, right, bottom))
        basewidth = 200    
        wpercent = (basewidth/float(im1.size[0]))
        hsize = int((float(im1.size[1])*float(wpercent)))
        img = im1.resize((basewidth,hsize), Image.ANTIALIAS)
        greyScaled = img.convert('L')    
        nameGrey = 'greyScaled' + filename 
        newpicGrey = imwrite(nameGrey, greyScaled, ".jpg")    


#Dataset creation section. will do it by now on the fly
variantes = ['Moca_10_11', '17_18', 'Cata']
nomes = []
labels = []
for variante in variantes:        
    filesVar = os.listdir('./amostrasGow/' + variante)
    for arquivo in filesVar:
        nomes.append(arquivo)
        labels.append(variante)
    
directory = './amostrasGow/'
batch_size = 32
img_height = 200
img_width = 200
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  directory,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  directory,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(200, 200, 3)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
class_names = train_ds.class_names
AUTOTUNE = tf.data.AUTOTUNE

#train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
#val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=3
)
#test_loss, test_acc = model.evaluate(val_ds,  labels, verbose=2)

#print('\nTest accuracy:', test_acc)

print(labels)
print(tf.__version__)
print(tfds.__version__)