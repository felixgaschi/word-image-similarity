"""
Script that generates a dataset with feature maps of word image
from a dataset of word image
"""

from models import *
import argparse
import os
from PIL import Image
import numpy as np
import torch
from shutil import copyfile
from tqdm import tqdm
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from embedding.dataset import WordImageLoader
from embedding.models import *

# TODO try with larger batch
# TODO try DataParallel


if __name__ == "__main__":

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


    def export_embedding(original, model, layer, channel, output, fname):
        embeddings = [] 

        def get_output(m, i, o):
            embeddings.append(o.detach())

        h = layer.register_forward_hook(get_output)

        model(original)

        h.remove()

        embedding = embeddings[0][:, channel]
        new = transforms.ToPILImage(mode="L")(embedding.cpu())

        new.save(os.path.join(output, fname))



    if args.model == "resnet50":
        model = resnet50
        layer = get_layer(model, args.layer)[0]
    else:
        raise ValueError("{} is a not a suitable value for option model".format(args.model))

    if use_cuda:
        print('Using GPU')
        model.cuda(args.gpu)
        model.share_memory()
    else:
        print('Using CPU')

    images = [f for f in os.listdir(args.input) if f[-4:] == ".png"]
    others = [f for f in os.listdir(args.input) if f[-4:] != ".png"]

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    model.eval()
    
    dataloader = torch.utils.data.DataLoader(
        WordImageLoader(
            args.input,
            transform=transforms.Compose([
                transforms.Resize((int(40 * args.factor), int(100 * args.factor))),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.cat((x, x, x), 0))
            ])
        ),
        batch_size=1, shuffle=False, num_workers=args.nb_workers
    )

    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        if use_cuda:
            data = data.cuda(args.gpu)
        
        export_embedding(data, model, layer, args.channel, args.output, images[i])

    for o in others:
        copyfile(os.path.join(args.input, o), os.path.join(args.output, o))


