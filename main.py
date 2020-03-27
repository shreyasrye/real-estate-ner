#This is a program that can detect titles and headers in images of webpages or text.

import pytesseract
from matplotlib import pyplot as plt
from preprocessimage import *
import spacy



IMG_DIR = '/Users/shreyasrai/Documents/pictures/'
pi = ProcessedImage()

#identify characters in the given images
def readtext():
    custom_config = r'--oem 3 --psm 6'

    name = input("Enter Image name: ")
    img = cv2.imread(IMG_DIR + name)

    gray = pi.get_grayscale(img)
    thresh = pi.thresholding(gray)
    opening = pi.opening(gray)
    canny = pi.canny(gray)

    text = pytesseract.image_to_string(img)
    return text



# Plot original image
def plotImage():
    image = cv2.imread(IMG_DIR + readtext().name)
    b, g, r = cv2.split(image)
    rgb_img = cv2.merge([r, g, b])
    plt.imshow(rgb_img)
    plt.title('ORIGINAL IMAGE')
    plt.show()


readtext()
#plotImage()


# nlp = spacy.load('en')
# doc = nlp(readtext())
#
# # Create list of word tokens
# token_list = []
# for token in doc:
#     token_list.append(token.text)
# print(token_list)

