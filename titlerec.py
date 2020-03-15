#This is a program that can detect titles and headers in images of webpages or text.
from PIL import Image
import cv2
import pytesseract

sampleURL = '/Users/shreyasrai/Documents/pictures/breakingnews.png'

def readtext(imgurl):
    img = cv2.imread(imgurl)
    text = pytesseract.image_to_string(img)
    print( text)


readtext(sampleURL)
