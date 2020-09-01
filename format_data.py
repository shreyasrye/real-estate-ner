"""
Extract text from the training images and train a model to recognize
named entities based on the data.
"""
import glob
import pytesseract
import json
import cv2
from PIL import Image as IMG
from doccano_transformer.datasets import NERDataset
from doccano_transformer.utils import read_jsonl


# For Windows only - setting the tesseract path
pytesseract.pytesseract.tesseract_cmd = \
    r'C:\Users\Shreyas\AppData\Local\Tesseract-OCR\tesseract.exe'


class Image:
    text = ""

    def __init__(self):
        self._text = ""

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, set_string):
        self._text = set_string

    def set_image_dpi(self, file_name):
        im = IMG.open(file_name)
        im.save("test-600.png", dpi=(300, 300))


class WebAnnoFormatter:
    train_data = []
    Sentence = []
    data = {}
    sentences_list = []

    def __init__(self):
        # Read output json file from WebAnno (Annotation tool)
        with open('webanno_annotations/image_text.json') as data_file:
            self.data = json.load(data_file)

    def fill_sentences_ls(self):
        for sent_borders in self.Sentence:
            tmp_sent_string = ""
            for letter in range(sent_borders["begin"], sent_borders["end"]):
                tmp_sent_string += (self.data['_referenced_fss']['12']['sofaString'][letter])
            self.sentences_list.append(tmp_sent_string)

    def format_json(self):
        # Extract entity start/ end positions and names
        # nlp = spacy.blank('en', disable=["tagger", "ner"])
        ent_loc = self.data['_views']['_InitialView']['NamedEntity']
        ent_list = []
        for sl in range(len(self.Sentence)):
            ent_list_sen = []
            for el in range(len(ent_loc)):
                try:
                    if ent_loc[el]['begin'] >= self.Sentence[sl]['begin'] and ent_loc[el]['end'] <= self.Sentence[sl]['end']:
                        # subtract entity location with sentence beginning as webanno generate data (doc as a whole)
                        ent_list_sen.append(
                            [(ent_loc[el]['begin'] - self.Sentence[sl]['begin']),
                             (ent_loc[el]['end'] - self.Sentence[sl]['begin']),
                             ent_loc[el]['value']])
                except KeyError:
                    ent_loc[el]['begin'] = 0

                    if ent_loc[el]['begin'] >= self.Sentence[sl]['begin'] and ent_loc[el]['end'] <= self.Sentence[sl]['end']:
                        # subtract entity location with sentence beginning as webanno generate data (doc as a whole)
                        ent_list_sen.append(
                            [(ent_loc[el]['begin'] - self.Sentence[sl]['begin']),
                             (ent_loc[el]['end'] - self.Sentence[sl]['begin']),
                             ent_loc[el]['value']])
            ent_list.append(ent_list_sen)
            # Fill value to a dictionary
            ent_dic = {'entities': ent_list[-1]}
            # Prepare final training data
            self.train_data.append((self.sentences_list[sl], ent_dic))

    def fill_train_data(self):

        # Extract Sentence start/ end positions
        self.Sentence = self.data['_views']['_InitialView']['Sentence']
        # Set first sentence starting position 0
        self.Sentence[0]['begin'] = 0

        # extract original sentences:
        self.fill_sentences_ls()

        # Prepare spacy formatted training data
        self.format_json()


def read_text(empty_ls):
    """ Use opencv and tesseract to extract data from the files and preprocesses images for accuracy"""
    config = '-l eng --oem 1 --psm 3'
    for file in glob.iglob(r'training_imgs/*.png'):
        im = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img_obj = Image()
        # Denoising images for better accuracy
        im = cv2.fastNlMeansDenoising(im, 5, 7, 21)
        # Binarize image
        (thresh, im_bw) = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        thresh = 127
        cv2.imwrite(file, im_bw)
        img_obj.text = pytesseract.image_to_string(im, config=config)
        empty_ls.append(img_obj)
    return empty_ls


def write_text(text_ls):
    """ Write the extracted text into a file for annotating"""
    file = open(r"image_text.txt", "a")
    for i in range(len(text_ls)):
        file.write(text_ls[i].text)
    file.close()


def main():
    # Using WebAnno for annotating
    df = WebAnnoFormatter()
    read_text(df.train_data)
    write_text(df.train_data)
    df.fill_train_data()

    # Using doccano
    dataset = read_jsonl(filepath='example.jsonl', dataset=NERDataset, encoding='utf-8')
    dataset.to_conll2003(tokenizer=str.split)
    dataset.to_spacy(tokenizer=str.split)


if __name__ == '__main__':
    main()
