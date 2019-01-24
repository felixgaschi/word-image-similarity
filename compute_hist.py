import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse, os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generating matching dataset for MNIST')

    parser.add_argument("--data", type=str, default="../dataset", metavar="I",
                        help="directory for the data")

    args = parser.parse_args()


    fnames = [f for f in os.listdir(args.data) if f[-4:] == ".png"]
    print(fnames)

    hist = np.zeros((256,))
    n = 0

    for fname in tqdm(fnames):
        img = Image.open(os.path.join(args.data, fname))

        arr = np.array(img)
        
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                hist[int(arr[i, j])] += 1
                n += 1
        
    hist /= n

    np.savetxt(os.path.join(args.data, "history.txt"), hist)
