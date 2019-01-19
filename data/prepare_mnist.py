from mlxtend.data import loadlocal_mnist
import argparse, os
from PIL import Image
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generating matching dataset for MNIST')

    parser.add_argument("--data", type=str, default="../mnist", metavar="I",
                        help="directory for the data")
    parser.add_argument("--out", type=str, default="../dataset-mnist", metavar="O")

    args = parser.parse_args()

    X, y = loadlocal_mnist(
        images_path=os.path.join(args.data, "train-images-idx3-ubyte"), 
        labels_path=os.path.join(args.data, "train-labels-idx1-ubyte"))
    X_test, y_test = loadlocal_mnist(
        images_path=os.path.join(args.data, "t10k-images-idx3-ubyte"), 
        labels_path=os.path.join(args.data, "t10k-labels-idx1-ubyte"))

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    with open(os.path.join(args.out, "words.txt"), "w") as f:
        f.write("")
    
    for i in tqdm(range(X.shape[0])):
        img = Image.fromarray(X[i].reshape(28,28))

        img.save(os.path.join(args.out, "word-{:06d}.png".format(i + 1)))

        with open(os.path.join(args.out, "words.txt"), "a") as f:
            f.write("{}\n".format(y[i]))
    
    for i in tqdm(range(X_test.shape[0])):
        img = Image.fromarray(X_test[i].reshape(28,28))

        img.save(os.path.join(args.out, "word-{:06d}.png".format(X.shape[0] + i + 1)))

        with open(os.path.join(args.out, "words.txt"), "a") as f:
            f.write("{}\n".format(y_test[i]))
