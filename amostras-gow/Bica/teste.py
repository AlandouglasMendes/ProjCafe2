# Importing Image class from PIL module
from PIL import Image
import os

from imageio import imread, imwrite
 
# Opens a image in RGB mode

directory = r'/home/alandouglas/ProjCafe2/amostras-gow/Bica/'
dirPath = '/home/alandouglas/ProjCafe2/amostras-gow/Bica/'
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

    

 
# Cropped image of above dimension
# (It will not change original image)



#greyScaled.save('somepicgs.jpg')