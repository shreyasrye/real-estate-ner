import spacy
import plac
from spacy.matcher import PhraseMatcher
from pathlib import Path
import random

PRICES = ["$299,000", "$285,000", "$267,900", "$263,000", "$599,000", "$1,800,000", "$1 ,8/72,000", "$2,350,000",
          "$749,000", "$525,000", "$279,000", "$69,500,000", "$39, 900, 000", "$1,298,000", "$179,900", "$419,000",
          "$2,040,000"]


def offseter(lbl, doc, matchitem):
    o_one = len(str(doc[0:matchitem[1]]))
    subdoc = doc[matchitem[1]:matchitem[2]]
    o_two = o_one + len(str(subdoc))
    return (o_one, o_two, lbl)


def phrase_match(nlp, lbl, label_ls):
    label = lbl
    matcher = PhraseMatcher(nlp.vocab)
    for i in label_ls:
        matcher.add(label, None, nlp(i))
    # Gather training data
    res = []
    to_train_ents = []
    with open('image_text.txt') as it:
        line = True
        while line:
            line = it.readline()
            mnlp_line = nlp(line)
            matches = matcher(mnlp_line)
            res = [offseter(label, mnlp_line, x) for x in matches]
            to_train_ents.append((line, dict(entities=res)))
    # Train the recognizer
    optimizer = nlp.begin_training()
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes): # only train the ner
        for itn in range(20):
            losses = {}
            random.shuffle(to_train_ents)
            for item in to_train_ents:
                nlp.update([item[0]], [item[1]], sgd=optimizer, drop=0.35, losses=losses)


def main():
    nlp = spacy.load('en')
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    else:
        ner = nlp.get_pipe('ner')

    phrase_match(nlp, 'PRICE', PRICES)


if __name__ == '__main__':
    main()
