import os
from tqdm import tqdm
import pickle as pk
import numpy as np

def register_pairs():
    root = "../preprocessed"
    jump_equal = True
    split = 0.7

    images = [os.path.join(root, f) for f in os.listdir(root) if f[-4:] == ".png"]

    assert len(images) > 0, "The data folder should contain png images"

    assert "words.txt" in os.listdir(root), "The data folder should contain 'words.txt'"

    words = []

    with open(os.path.join(root, "words.txt"), "r") as f:
        words = [w.strip() for w in f.readlines()]

    train = []
    val = []

    pbar = tqdm(total=len(words) * len(words))
    for i, w1 in enumerate(words):
        a = os.path.join(root, "word-{:06d}.png".format(i + 1))
        if a not in images:
            pbar.update(len(words))
            print("WARNING: data root doesn't contain 'word-{:06d}.png'".format(i + 1))
            continue
        for j, w2 in enumerate(words):
            if i == j and jump_equal:
                pbar.update(1)
                continue
            b = os.path.join(root, "word-{:06d}.png".format(j + 1))
            if b not in images:
                print("WARNING: data root doesn't contain 'word-{:06d}.png'".format(j + 1))
                continue
            y = 1 if w1 == w2 else 0
            if np.random.random() < split:
                train.append(((a, b), y))
            else:
                val.append(((a, b), y))
            pbar.update(1)
    pbar.close()

    with open(os.path.join(root, "train.pk"), "wb") as f:
        pk.dump(train, f)

    with open(os.path.join(root, "eval.pk"), "wb") as f:
        pk.dump(val, f)


if __name__ == "__main__":
    register_pair()
