"""
Extract text from the training images and train a model to recognize
named entities based on the data.
"""
import glob
import pytesseract

import cv2
import random
import warnings
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding

# For Windows only - setting the tesseract path
pytesseract.pytesseract.tesseract_cmd = \
    r'C:\Users\shreyasrai\AppData\Local\Tesseract-OCR\tesseract.exe'

# Will later become a list of tuples of real estate data
TRAIN_DATA = []


def extract_text(empty_ls):
    """ Use opencv and tesseract to extract data from the files"""
    for file in glob.iglob('training_imgs/*.png'):
        img = cv2.imread(file)

        empty_ls.append(pytesseract.image_to_string(img))


def main(model=None, output_dir=None, n_iter=100):
    extract_text(TRAIN_DATA)
    nlp = spacy.blank("en")


if __name__ == '__main__':
    main()
