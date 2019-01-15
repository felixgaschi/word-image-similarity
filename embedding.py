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
import torchvision.models as models
import torch.nn as nn
import torchvision.models as models
import torch
import torch.nn as nn
import torch.utils.data as data
import os

def grey_pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert("L")

class WordImageLoader(data.Dataset):

    def __init__(self, root, loader=grey_pil_loader, transform=None, extension=".png"):
        self.root = root
        self.loader = grey_pil_loader
        self.transform = transform

        self.filenames = [os.path.join(root, f) for f in os.listdir(root) if f[-len(extension):] == extension]

    def __getitem__(self, index):
        fname = self.filenames[index]
        x = self.loader(fname)
        if self.transform is not None:
            x = self.transform(x)
        return x
    
    def __len__(self):
        return len(self.filenames)


def get_embedding(model, layer, channel, tensor):
    """
    return an embedding for a given image stored in a pytorch tensor
    computed with the result of a given channel of a given layer of 
    the model
    """
    model.eval()

    tensor = torch.cat((tensor.unsqueeze(0), tensor.unsqueeze(0), tensor.unsqueeze(0)), 1)

    embedding = []
    def get_output(m, i, o):
        embedding.append(o.detach())
    
    h = layer.register_forward_hook(get_output)

    model(tensor)

    h.remove()

    return embedding[0][:, channel]





def get_layer(model, i, counter=0):
    """
    Returns the i-th conv2d layer of a model
    If the architecture has nested layers (conv2d inside sequential)
    it returns the first layer by depth-first search
    """
    res = None
    for layer in model.children():
        if res is not None:
            return res, counter
        if isinstance(layer, nn.modules.conv.Conv2d):
            if counter == i:
                return layer, -1
            counter += 1
        else:
            res, counter = get_layer(layer, i, counter=counter)
    return res, counter

resnet50 = models.resnet50(pretrained=True)
resnet50.avgpool = nn.AdaptiveAvgPool2d(1)

resnet50_layer = get_layer(resnet50, 44)[0]
resnet50_channel = 212


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


