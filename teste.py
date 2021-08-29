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

import amostrasGow.coffeSingleDataset as ds


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
variantes = ['Bica', 'Moca_10_11', '17_18', 'Cata']
nomes = []
labels = []
for variante in variantes:        
    filesVar = os.listdir('./amostrasGow/' + variante)
    for arquivo in filesVar:
        nomes.append(arquivo)
        labels.append(variante)
    print(filesVar)
    print(labels)
    print('--------------------------')
directory = './amostrasGow/'
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
print('info:  ')
print(datajob.info())
print(datajob.cardinality().numpy())

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(200, 200)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


print(labels)
print(tf.__version__)
print(tfds.__version__)