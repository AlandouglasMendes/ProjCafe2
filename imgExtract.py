from __future__ import print_function
# Importing Image class from PIL module, nd module PIL
from PIL import Image
import PIL

from datetime import datetime

import keras
#importing lib numpy
import numpy as np

#os allow us to manipulate dir, folders, files
import os



#lib for dir manipulating
import pathlib

#image manipulation
from imageio import imread, imwrite



#dir is the path to a diretory where images will be processed, the result being output in the same dir
def procesImage(dir):
    directory = r'./amostrasGow/' + dir + '/' 
    print(directory)
    dirPath = directory
    counter = 54
    for filename in os.listdir(directory):        
        if filename.startswith('2'):
            imName = directory +  filename
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
            nameGrey = 'greyScaled17_18' + str(counter) + '.jpg'
            counter = counter + 1
            newpicGrey = imwrite(nameGrey, greyScaled, ".jpg")
            idx = counter - 52
            print('img ' + str(idx))
procesImage("17_18")