import logging

import os

import re

import os.path

from pprint import pprint

import random
import requests
from PIL import Image

import tensorflow as tf

from imageio import imread, imwrite

from telegram import Update, ForceReply

import matplotlib.pyplot as plt

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import numpy as np

from datetime import datetime

new_model = tf.keras.models.load_model('precisao=66')

# Check its architecture
new_model.summary()

#Storing date and time as a string for later use in code creating logs etcs
nowRaw = datetime.now()
string = str(nowRaw)
pattern = re.compile('\W')
now = re.sub(pattern, '', string)
print(now)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)



logger = logging.getLogger(__name__)
def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    update.message.reply_markdown_v2(
        fr'Hi {user.mention_markdown_v2()} I am the coffee bot inspector, send me a picture of a coffee bean and I\'ll try to gues its quality\!',
    )

def echo(update: Update, context: CallbackContext) -> None:
    """Takes an image of a bean as an input, output is the estimated quality of the bean"""  
    #Setting image name  
    user = update.message.from_user
    
    fileId = update.message.document.file_id
    photo_file = update.message.document.get_file()
    
    
    photo_file.download('test.jpg')
    pic = Image.open('test.jpg') 
    width, height = pic.size
    left = (width/ 2.694) + 200
    top = (height / 5.694) + 300
    right = (width / 2.694) + 908
    bottom = (height / 5.694) + 1008  
    im1 = pic.crop((left, top, right, bottom))
    basewidth = 200    
    wpercent = (basewidth/float(im1.size[0]))
    hsize = int((float(im1.size[1])*float(wpercent)))
    img = im1.resize((basewidth,hsize), Image.ANTIALIAS)
    greyScaled = img.convert('L')
    nameGrey = now + '.jpg'   

    #creating folder to store testpics 
    os.mkdir('sessao=' + now)
    save_path = 'sessao=' + now 
    newpicGrey = imwrite(save_path + nameGrey, greyScaled, ".jpg")
    imageTest = Image.open(save_path + nameGrey)
    imageTest = (np.expand_dims(imageTest,0))    
    prediction = new_model.predict(imageTest)
    solution = np.argmax(prediction[0])
    class_names = ['17_18', 'Cata', 'Moca_10_11']
    guess = class_names[solution]   
    print(guess)     
              
  
    #log creation for the session
    logline = "palpite da imagem foi " + guess
    name_of_file = 'log' + now + ".txt"
    completeName = os.path.join(save_path, name_of_file )
    f = open(completeName, "w")
    f.close()
    f = open(completeName, "a")
    f.write(logline)
    f.write('\n\n\n\n')
    f.write('-------------------------------------------------------------------------------------------\n')
    f.close()
    #Showing user result
    update.message.reply_text(guess)
    

    

def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')


def main() -> None:
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    updater = Updater("1836757640:AAG1K5XHn8ehGRXRIYTkycZPNU1KfAnmLVc")

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))

    # on non command i.e message - echo the message on Telegram
    dispatcher.add_handler(MessageHandler(Filters.document.mime_type("image/jpeg") & ~Filters.command, echo))

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()
