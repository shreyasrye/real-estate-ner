#This is a program that can detect titles and headers in images of webpages or text.

import re
import cv2
import pytesseract
from pytesseract import Output
from matplotlib import pyplot as plt

IMG_DIR = 'pictures/'
sampleURL = '/Users/shreyasrai/Documents/pictures/breakingnews.png'

#identify characters in the given images
def readtext(imgurl):
    img = cv2.imread(imgurl)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(img)
    print( text)

# Plot original image
def plotImage():
    image = cv2.imread(IMG_DIR + 'aurebesh.jpg')
    b, g, r = cv2.split(image)
    rgb_img = cv2.merge([r, g, b])
    plt.imshow(rgb_img)
    plt.title('AUREBESH ORIGINAL IMAGE')
    plt.show()



readtext(sampleURL)
