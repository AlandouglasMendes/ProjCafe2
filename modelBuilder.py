from __future__ import print_function
# Importing Image class from PIL module, nd module PIL
from PIL import Image
import PIL

import os.path


from datetime import datetime

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
nowRaw = datetime.now()
now = str(nowRaw)
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
  color_mode="grayscale",
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  directory,
  validation_split=0.2,
  color_mode="grayscale",
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
num_classes = 3

model = tf.keras.models.Sequential([
  keras.layers.Flatten(input_shape=(200, 200)),
  keras.layers.LeakyReLU(alpha=0.1),
  keras.layers.LeakyReLU(alpha=0.1),
  keras.layers.Flatten(input_shape=(200, 200)),
  keras.layers.LeakyReLU(alpha=0.1),
  keras.layers.LeakyReLU(alpha=0.1),
  keras.layers.Flatten(input_shape=(200, 200)),
  keras.layers.LeakyReLU(alpha=0.1),
  
  keras.layers.Dense(3, activation='softmax')
])
class_names = train_ds.class_names
AUTOTUNE = tf.data.AUTOTUNE



model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=20
)

logstring = ["Dataset treinado em: " + now + "\n"]
acertos = 0
total = 0
for classification in class_names:
  for filename in os.listdir(directory + classification):
    if filename.endswith('.jpg'):
      imageTest = Image.open(r'./amostrasGow/'+ classification + '/' + filename)
      imageTest = (np.expand_dims(imageTest,0))    
      prediction = model.predict(imageTest)
      solution = np.argmax(prediction[0])
      nomeResult = class_names[solution]
      logstring.append(filename + ': ' + class_names[solution]  + '\n')
      total = total + 1
      if class_names[solution] == classification:        
        acertos = acertos + 1
indiceAcerto = acertos / total
print('acertos: ' +  str(acertos) + '|||' + 'total: ' + str(total) + '\n')
print('acertos: ' + str(acertos / total) + '!' + '\n')
menu = input("1 para salvar modelo/log, 2 para descartar:\n")
indiceAcertStr = str(indiceAcerto)
indAcertoDecimal = indiceAcertStr[0:5]
if menu == "1":
  os.mkdir('precisao=' + indAcertoDecimal[2:4])
  save_path = 'precisao=' + indAcertoDecimal[2:4]  
  logstringJoined = "".join(logstring)
  name_of_file = 'log' + now + ".txt"
  completeName = os.path.join(save_path, name_of_file )
  f = open(completeName, "w")
  f.write(logstringJoined)
  f.write('\n\n\n\n')
  f.write('-------------------------------------------------------------------------------------------\n')
  f.write("acertos/erros no conjunto total de imagens:  acertou " + str(acertos) +" de um total de " + str(total) + " imagens, cerca de " + indAcertoDecimal + '\n' + 
          "por cento.") 
  f.close()
  model.save(save_path)
if menu == "2":
  print("não salvo")      
    
print(class_names)
