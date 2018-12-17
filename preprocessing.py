"""
Script for creating a set of words with their transcription from the GW dataset (http://ciir.cs.umass.edu/downloads/gw/)
"""

import argparse
import os, sys
from PIL import Image
import re
import scipy.misc
from tqdm import tqdm
import numpy as np
import string

parser = argparse.ArgumentParser(description='Word Image Similarity preprocessing script')

parser.add_argument("--input", type=str, default="../gw_20p_wannot", metavar="I",
                    help="directory for the input data")
parser.add_argument("--out", type=str, default="../preprocessed", metavar="O",
                    help="directory for the ouptut data")

args = parser.parse_args()


# Get the names of the image files
imageFnames = []

with open(os.path.join(args.input, "file_order.txt"), "r") as f:
    imageFnames = f.readlines()

imageFnames = [f.strip() for f in imageFnames]

# Get the names of the files containing the boxes coordinates
boxFnames = [f[:-4] + "_boxes.txt" for f in imageFnames]

# Get the list of words
words = []

with open(os.path.join(args.input, "annotations.txt"), "r") as f:
    words = f.readlines()

words = [w.strip() for w in words]

boxes = []
i = 0
j = -1
k = 0

# Create output directory if it doesn't exist
if not os.path.exists(args.out):
    os.makedirs(args.out)

image = None
new_words = []

pbar = tqdm(total=len(words))

while i < len(words):
    pbar.update(1)

    if k == len(boxes): 
        boxes = []
        j += 1
        k = 0

    if boxes == []:
        if j >= len(boxFnames):
            print("ERROR: the end of box files was reached before the end of word list. j={}; i={}".format(j, i))
            break
        image = np.array(Image.open(os.path.join(args.input, imageFnames[j])))
        with open(os.path.join(args.input, boxFnames[j])) as f:
            boxes_str = f.readlines()
        coords = [[float(i) for i in re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", s)] for s in boxes_str]
        boxes = [c[:4] for c in coords if len(c) >= 4]
    
    #if re.match('.*\d+', words[i]):
    #    i += 1
    #    k += 1
    #   continue

    new_words.append(words[i]) # re.sub(r'[^\w\s]','',words[i])

    a, b = image.shape[0], image.shape[1]
    if len(boxes[k]) < 4:
        print(i, j, k, boxes[k])
    x1, x2, y1, y2 = int(b * boxes[k][0]), int(b * boxes[k][1]), int(a * boxes[k][2]), int(a * boxes[k][3])

    res = image[y1:y2, x1:x2]

    im = Image.fromarray(res)
    im.save(os.path.join(args.out, "word-{:06d}.png".format(len(new_words))))

    k += 1
    i += 1

with open(os.path.join(args.out, "words.txt"), "w") as f:
    for w in new_words:
        f.write(w + '\n')
