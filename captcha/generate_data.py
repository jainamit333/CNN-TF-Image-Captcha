import os
from random import randint
from captcha.image import ImageCaptcha


print("Hi I will start working")

directory_path = "images/"
image = ImageCaptcha(width=60, height=60)

if not os.path.exists(directory_path):
    os.makedirs(directory_path)
    print("create directory")




numberOfImages = 500

for i in range(numberOfImages):

    number = randint(0,9)
    fileName = 'images/test/' + str(i)+ '_' + str(number) + '.png'
    image.write(str(number), fileName)
