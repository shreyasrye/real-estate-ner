"""
Extract text from the training images and train a model to recognize
named entities based on the data.
"""
import glob
import pytesseract
import json
import cv2
import spacy
from spacy.gold import GoldCorpus, minibatch, biluo_tags_from_offsets, tags_to_entities

# For Windows only - setting the tesseract path
pytesseract.pytesseract.tesseract_cmd = \
    r'C:\Users\Shreyas\AppData\Local\Tesseract-OCR\tesseract.exe'

TRAIN_DATA = []


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


def format_json(train_data, anno_percentage: float):

    # Read output json file from WebAnno (Annotation tool)
    with open('webanno_annotations/image_text.json') as data_file:
        data = json.load(data_file)
        print(type(data))

    # # Extract original sentences
    # sentences_list = data['_referenced_fss']['12']['sofaString'].split('\r\n')


    # Extract entity start/ end positions and names
    ent_loc = data['_views']['_InitialView']['NamedEntity']

    # Extract Sentence start/ end positions
    Sentence = data['_views']['_InitialView']['Sentence']
    # Set first sentence starting position 0
    Sentence[0]['begin'] = 0

    # extract original sentences:
    sentences_list = []
    for sent_borders in Sentence:
        tmp_sent_string = ""
        for letter in range(sent_borders["begin"], sent_borders["end"]):
            tmp_sent_string += (data['_referenced_fss']['12']['sofaString'][letter])
        sentences_list.append(tmp_sent_string)

    # Prepare spacy formatted training data
    ent_list = []
    for sl in range(len(Sentence)):
        ent_list_sen = []
        for el in range(len(ent_loc)):
            try:
                if ent_loc[el]['begin'] >= Sentence[sl]['begin'] and ent_loc[el]['end'] <= Sentence[sl]['end']:
                    # subtract entity location with sentence beginning as webanno generate data (doc as a whole)
                    ent_list_sen.append(
                        [(ent_loc[el]['begin'] - Sentence[sl]['begin']), (ent_loc[el]['end'] - Sentence[sl]['begin']),
                         ent_loc[el]['value']])
            except KeyError:
                ent_loc[el]['begin'] = 0

                if ent_loc[el]['begin'] >= Sentence[sl]['begin'] and ent_loc[el]['end'] <= Sentence[sl]['end']:
                    # subtract entity location with sentence beginning as webanno generate data (doc as a whole)
                    ent_list_sen.append(
                        [(ent_loc[el]['begin'] - Sentence[sl]['begin']), (ent_loc[el]['end'] - Sentence[sl]['begin']),
                         ent_loc[el]['value']])

        ent_list.append(ent_list_sen)
        # Create blank dictionary
        ent_dic = {'entities': ent_list[-1]}
        # Fill value to the dictionary
        # Prepare final training data
        train_data.append((sentences_list[sl], ent_dic))


def read_text(empty_ls):
    """ Use opencv and tesseract to extract data from the files"""
    for file in glob.iglob('training_imgs/*.png'):
        img = cv2.imread(file)
        img_obj = Image()
        img_obj.text = pytesseract.image_to_string(img)
        empty_ls.append(img_obj)
    return empty_ls


def write_text(text_ls):
    """ Write the extracted text into a file for annotating"""
    file = open(r"image_text.txt", "a")
    for i in range(len(text_ls)):
        file.write(text_ls[i].text)
    file.close()


def main():
    # read_text(TRAIN_DATA)
    # write_text(TRAIN_DATA)
    nlp = spacy.blank('en', disable=["tagger", "ner"])
    format_json(TRAIN_DATA, 0.375)
    for i in TRAIN_DATA:
        print(i)


if __name__ == '__main__':
    main()
