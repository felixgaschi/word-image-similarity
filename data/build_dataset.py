"""
Build dataset from list of pairs ids
"""
import argparse, os, csv
from PIL import Image
from tqdm import tqdm
import numpy as np

def build_dataset(args):

    # Create output directory if it doesn't exist
    if not os.path.exists(os.path.join(args.out, "train", "same")):
        os.makedirs(os.path.join(args.out, "train", "same"))

    if not os.path.exists(os.path.join(args.out, "train", "different")):
        os.makedirs(os.path.join(args.out, "train", "different"))

    if not os.path.exists(os.path.join(args.out, "eval", "same")):
        os.makedirs(os.path.join(args.out, "eval", "same"))

    if not os.path.exists(os.path.join(args.out, "eval", "different")):
        os.makedirs(os.path.join(args.out, "eval", "different"))

    def read_csv(filename):
        with open(filename) as f:
            reader = csv.reader(f)
            for row in reader:
                yield row


    n_true = 0
    for i in read_csv(os.path.join(args.data, "true-pairs.csv")):
        n_true += 1

    n_false = 0
    for i in read_csv(os.path.join(args.data, "false-pairs.csv")):
        n_false += 1

    if args.limit is not None:
        n_true = min(args.limit, n_true)
        n_false = min(args.limit, n_false)

    ratio = 0.70 if args.ratio is None else args.ratio
    n_true_train = int(ratio * n_true)
    n_false_train = int(ratio * n_false)

    pbar = tqdm(total=n_true + n_false, position = 0)
    for i, row in enumerate(read_csv(os.path.join(args.data, "true-pairs.csv"))):

        if i == n_true:
            break

        a = np.array(Image.open(os.path.join(args.data, "word-" + row[0] + ".png")).resize((100,40)))
        b = np.array(Image.open(os.path.join(args.data, "word-" + row[1] + ".png")).resize((100,40)))

        res = np.zeros((40, 100, 3), 'uint8')
        res[:,:,0] = a
        res[:,:,1] = b

        im = Image.fromarray(res)
        
        if i < n_true_train:
            im.save(os.path.join(args.out, "train", "same", "{:012d}.png".format(i)))
        else:
            im.save(os.path.join(args.out, "eval", "same", "{:012d}.png".format(i)))
        pbar.update(1)

    for i, row in enumerate(read_csv(os.path.join(args.data, "false-pairs.csv"))):

        if i == n_false:
            break

        a = np.array(Image.open(os.path.join(args.data, "word-" + row[0] + ".png")).resize((100,40)))
        b = np.array(Image.open(os.path.join(args.data, "word-" + row[1] + ".png")).resize((100,40)))

        res = np.zeros((40, 100, 3), "uint8")
        res[:,:,0] = a
        res[:,:,1] = b

        im = Image.fromarray(res)
        
        if i < n_false_train:
            im.save(os.path.join(args.out, "train", "different", "{:012d}.png".format(i)))
        else:
            im.save(os.path.join(args.out, "eval", "different", "{:012d}.png".format(i)))
        pbar.update(1)
    pbar.close()

if __name__ = "__main__":
    parser = argparse.ArgumentParser(description='Word Image Similarity script for constructing ground truth')

    parser.add_argument("--data", type=str, default="../preprocessed", metavar="I",
                        help="directory for the data")
    parser.add_argument("--out", type=str, default="../pair-dataset", metavar="O")
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--ratio", type=int, default=None)

    args = parser.parse_args()

    build_dataset(args)
