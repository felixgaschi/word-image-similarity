"""
Script that generates a dataset with feature maps of word image
from a dataset of word image
"""

from models import *
from get_embedding import *
import argparse
import os
from PIL import Image
import numpy as np
import torch
from shutil import copyfile
from tqdm import tqdm
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='Generates a dataset with a feature map instead of the word image')

parser.add_argument("--input", type=str, default="../preprocessed", metavar="I",
                    help="directory for the data")
parser.add_argument("--output", type=str, default="../feature_maps", metavar="O",
                    help="limit number of pairs")
parser.add_argument("--model", type=str, default="resnet50")
parser.add_argument("--layer", type=int, default=44)
parser.add_argument("--channel", type=int, default=212)
parser.add_argument("--factor", type=int, default=20)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--nb-workers", type=int, default=1)

args = parser.parse_args()

use_cuda = torch.cuda.is_available()


if args.model == "resnet50":
    model = resnet50
    layer = get_layer(model, args.layer)[0]
else:
    raise ValueError("{} is a not a suitable value for option model".format(args.model))

if use_cuda:
    print('Using GPU')
    model.cuda(args.gpu)
else:
    print('Using CPU')

images = [f for f in os.listdir(args.input) if f[-4:] == ".png"]
others = [os.path.join(args.input, f) for f in os.listdir(args.input) if f[-4:] != ".png"]

if not os.path.exists(args.output):
    os.makedirs(args.output)

for fname in tqdm(images):

    original = Image.open(os.path.join(args.input, fname))

    original = transforms.Resize((int(40 * args.factor), int(100 * args.factor)))(original)
    original = transforms.ToTensor()(original)

    embedding = get_embedding(model, layer, args.channel, original)

    new = transforms.ToPILImage(mode="L")(embedding)

    new.save(os.path.join(args.output, fname))

for o in others:
    copyfile(o, args.output)
