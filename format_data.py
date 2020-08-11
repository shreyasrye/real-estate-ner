"""
Extract text from the training images and train a model to recognize
named entities based on the data.
"""
import glob
import pytesseract
import json
import cv2


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


class DataFormatter:
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
    df = DataFormatter()
    read_text(df.train_data)
    write_text(df.train_data)
    df.fill_train_data()


if __name__ == '__main__':
    main()
