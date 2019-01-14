"""
Build a new dataset with random jittered images
"""

import argparse, os
import numpy as np
from tqdm import tqdm
from PIL import Image
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Word Image Similarity script for constructing transformed ground truth')

    parser.add_argument("--data", type=str, default="../gw_20p_wannot", metavar="I",
                        help="directory for the data")
    parser.add_argument("--out", type=str, default="../dataset", metavar="O")

    args = parser.parse_args()


def create_transformed_dataset(args):

    data_dir = args.data
    data_out = args.out

    # Get the names of the image files
    imageFnames = []

    with open(os.path.join(data_dir, "file_order.txt"), "r") as f:
        imageFnames = f.readlines()

    imageFnames = [f.strip() for f in imageFnames]

    # Get the names of the files containing the boxes coordinates
    boxFnames = [f[:-4] + "_boxes.txt" for f in imageFnames]

    words = []

    with open(os.path.join(data_dir, "annotations.txt"), "r", encoding = "iso-8859-1") as f:
        words = [w.strip() for w in f.readlines()]
    
    nb_transforms = [2 * words.count(w) for w in words]

    if not os.path.exists(data_out):
        os.makedirs(data_out)

    boxes = []
    i = 0
    j = -1
    k = 0

    image = None
    new_words = []

    pbar = tqdm(total=np.sum(nb_transforms), position = 0)

    while i < len(words):

        if k == len(boxes): 
            boxes = []
            j += 1
            k = 0

        if boxes == []:
            if j >= len(boxFnames):
                print("ERROR: the end of box files was reached before the end of word list. j={}; i={}".format(j, i))
                break
            image = np.array(Image.open(os.path.join(data_dir, imageFnames[j])))
            with open(os.path.join(data_dir, boxFnames[j])) as f:
                boxes_str = f.readlines()
            coords = [[float(i) for i in re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", s)] for s in boxes_str]
            boxes = [c[:4] for c in coords if len(c) >= 4]

        new_words.append(words[i]) 

        a, b = image.shape[0], image.shape[1]
        x1, x2, y1, y2 = int(b * boxes[k][0]), int(b * boxes[k][1]), int(a * boxes[k][2]), int(a * boxes[k][3])
        Sx, Sy = 5, 5

        x1prim = x1 - Sx
        y1prim = y1 - Sy
        x2prim = x2 + Sx
        y2prim = y2 + Sy

        res = image[y1prim:y2prim, x1prim:x2prim]

        im = Image.fromarray(res)
        im.save(os.path.join(data_out, "word-{:06d}.png".format(len(new_words))))
        pbar.update(1)

        k += 1
        i += 1

    with open(os.path.join(data_out, "words.txt"), "w") as f:
        for w in new_words:
            f.write(w + '\n')

if __name__ == "__main__":
    create_transformed_dataset(args)
