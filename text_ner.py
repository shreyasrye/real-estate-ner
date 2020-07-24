"""Detects named entities in images of Zillow house listings."""
import spacy
from preprocessimage import *
import pytesseract
from matplotlib import pyplot as plt


class TextRecognition:
    IMG_DIR = ''
    name = ""

    def read_text(self):
        """ Identify characters in the given images"""
        text = ""
        while True:
            img_dir = input("Enter your image directory and name (Example: "
                            "/User/Documents/Pictures/img.jpg):\n")
            img = cv2.imread(img_dir + self.name)
            try:
                text = pytesseract.image_to_string(img)
            except TypeError:
                print("\nPlease enter a valid directory\n")
                continue
            else:
                break
        return text


def print_text(text):
    print(f"\nFULL TEXT:")
    print("--------------------------------------------------------------")
    print(f"\n\n\n{text}\n\n\n")
    print("NER")
    print("--------------------------------------------------------------")


def main():
    nlp = spacy.load("en_core_web_sm")
    t = TextRecognition()
    text = t.read_text()
    print_text(text)
    doc = nlp(text)
    print("Text                                               Start "
          "character          End "
          "Character            Label")
    print("----                                               -------------"
          "--          "
          "-------------            -----")
    for ent in doc.ents:
        e_text = ent.text.strip()
        print(f"{e_text:<50}{ent.start_char:<25}{ent.end_char:<25}"
              f"{ent.label_:<25}")


if __name__ == '__main__':
    main()


"""
-----sample run code-----

Enter your image directory and name (Example: /User/Documents/Pictures/img.jpg):
no

Please enter a valid directory

Enter your image directory and name (Example: /User/Documents/Pictures/img.jpg):
/Users/shreyasrai/Documents/samplepictures/amplification.jpg

FULL TEXT:
--------------------------------------------------------------



Amplification

Back in the days of swing and big bands, the poor guitar player had nothing more than a small, low-
powered amplifier to try and cut through the sound of all those horns. During the early 1960s, when
bands such as the Beatles were struggling to be heard against thousands of screaming fans, a new kind
of amp was born out of necessity, the high-powered 4x12 stack.

A guitar amp will usually consist of two elements:
equalization (see our “Setting the Tone” panels later in
the book), which is a combination of treble, middle, and
bass; and gain and volume. Your volume is the master for
the overall output, while the gain or drive control increases
or decreases the degree to which that is saturated with
an overdriven or distorted sound.

You may have come across the great valve/transistor
debate. Valve amps are the real deal and have plenty
of “mojo,” whether bright and clean like a Fender Twin
and Roland Jazz Chorus, or classically thumping like a

Marshall amplifier
Known for great tone, reliability, and
versatility, Marshall amps have been many
players’ favourites for decades.

Marshall. Transistor amps are usually as reliable as
a hammer in comparison, as valve amps need more
frequent servicing.

Classic valve amps need to be turned up to achieve
their overloaded sound. Until you're really sure you want
to stick with the electric guitar, transistors are fine, but
you will need at least 50 watts of transistor power if you
want to compete with even a nottoo-heavy drummer,
and at least 15-20 watts of valve power. (Valves sound
louder at the same rating.) An amp has pre-amp and
power-amp stages, and some affordable “valvestate”
amps go for a combination, with one stage featuring
valves and the other transistors.

For the bedroom, however, there are now some
great, no-frills valve practice amps, which, at around
5 watts, give you a great overdriven sound at home-
stereo volume.

Channel switching allows you to swap between two
or three “dirty” or “clean” (distorted or not) sounds in
the same song. (You may want to check if the footswitch
is included in the amp price.) Fender pioneered the
inclusion of tremolo and spring reverb in guitar amps,
but these days there’s also a growing range of digital-

modelling amps and combos, which include effects (and
even different amp and cabinet simulations). These
aren’t real analogue effects, but computer simulations
triggered by your playing. Nonetheless, they can be a
great way of checking out what effects can do, if you
don’t own a load, or a “multi-FX,” unit already.



NER
--------------------------------------------------------------
Text                                               Start character          End Character            Label
----                                               ---------------          -------------            -----
the days                                          23                       31                       DATE                     
the early 1960s                                   193                      208                      DATE                     
Beatles                                           233                      240                      GPE                      
thousands                                         277                      286                      CARDINAL                 
4x12                                              368                      372                      CARDINAL                 
two                                               418                      421                      CARDINAL                 
Setting the Tone                                  455                      471                      WORK_OF_ART              
a Fender Twin                                     902                      915                      WORK_OF_ART              
Roland Jazz Chorus                                920                      938                      PERSON                   
9                                                 1019                     1020                     CARDINAL                 
Marshall                                          1026                     1034                     PERSON                   
Marshall                                          1097                     1105                     PERSON                   
decades                                           1150                     1157                     DATE                     
Marshall                                          1160                     1168                     PERSON                   
Classic                                           1282                     1289                     FAC                      
at least 50 watts                                 1466                     1483                     QUANTITY                 
at least 15-20                                    1565                     1579                     CARDINAL                 
one                                               1750                     1753                     CARDINAL                 
5 watts                                           1905                     1912                     QUANTITY                 
two                                               2018                     2021                     CARDINAL                 
three                                             2025                     2030                     CARDINAL                 
these days                                        2246                     2256                     DATE                     
2                                                 2617                     2618                     CARDINAL                 
019                                               2620                     2623                     CARDINAL
"""
