"""
file for constructing ground truth for similarity
build a csv with true pairs (pair of different handwritting of same word)
build another csv file with three columns:
- id1: id of first word
- id2: id of second word
- similarity: similarity score (1 if same word, 0 otherwise)
"""

import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Word Image Similarity script for constructing ground truth')

parser.add_argument("--data", type=str, default="../preprocessed", metavar="I",
                    help="directory for the data")
parser.add_argument("--limit", type=int, default=500000, metavar="L",
                    help="limit number of pairs")

args = parser.parse_args()

words = []

with open(os.path.join(args.data, "words.txt")) as f:
    words = [w.strip() for w in f.readlines()]

if args.limit is None:
    limit = len(words) * len(words)
else:
    limit = args.limit

with open(os.path.join(args.data, "true-pairs.csv"), "w") as f:
    f.write("")

nb_true = 0
pbar = tqdm(total=limit)
get_out = False
for i, w in enumerate(words):
    for j, w2 in enumerate(words):
        if i != j and w == w2:
            if nb_true < limit // 2:
                nb_true += 1
                with open(os.path.join(args.data, "true-pairs.csv"), "a") as f:
                    f.write("{:06d},{:06d}\n".format(i + 1, j + 1))
                pbar.update(1)
            else:
                get_out = True
                break
        if get_out:
            break
if not get_out:

nb_false = limit - nb_true

with open(os.path.join(args.data, "false-pairs.csv"), "w") as f:
    f.write("")

i, j, k = 0, 0, 0
while i < nb_false:
    
    if k == len(words):
        j, k = j + 1, 0
    
    if j == len(words):
        print("The process has restarted. There might be the same data twice in the dataset")
        j, k = 0, 0
    
    if words[j] != words[k]:
        i += 1
        pbar.update(1)
        with open(os.path.join(args.data, "false-pairs.csv"), "a") as f:
            f.write("{:06d},{:06d}\n".format(j + 1, k + 1))
    
    k += 1
pbar.close()

print("Number of true pairs selected: ", nb_true)
