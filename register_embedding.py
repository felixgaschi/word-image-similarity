import os

from tqdm import tqdm

import numpy as np

from shutil import copyfile, rmtree

from PIL import Image

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets

from data import keep2chan

def transformation():
    return transforms.Compose([
        transforms.Resize((40, 100)),
        transforms.ToTensor(),
        transforms.Lambda(lambda img: torch.cat((img, img, img), 1)),
        transforms.Normalize(mean=[0.45, 0.45], std=[0.22, 0.22])
    ])


def get_empirical_gaussian(X, y, nb_classes):
    n = X.shape[0]
    d = X.shape[1]

    mu = np.zeros((nb_classes, d))
    sigma = np.zeros((nb_classes, d, d))
    pi = np.zeros((nb_classes,))
    
    for i in range(n):
        pi[int(y[i])] += 1
        mu[int(y[i])] += X[i]
    for i in range(nb_classes):
        mu[i] = mu[i] / pi[i]
    for i in range(n):
        sigma[int(y[i])] += np.outer(X[i] - mu[i], X[i] - mu[i])
    for i in range(nb_classes):
        sigma[i] = sigma[i] / pi[i]
    pi = pi / n

    return pi, mu, sigma


def get_gaussian_LLH(X, y, nb_classes):
    n = X.shape[0]
    d = X.shape[1]
    
    pi, mu, sigma = get_empirical_gaussian(X, y, nb_classes)

    LLH = 0
    for i in range(n):
        j = int(y[i])
        LLH += np.log(pi[j]) \
                - (d / 2) * np.log(2 * np.pi) \
                - 0.5 * np.log(np.det(sigma[j])) \
                - 0.5 * (X[i] - mu[j]).T.dot(sigma[j]).dot(X[i] - mu[j])
    return LLH


if __name__ == "__main__":
    # TODO load data
    IMAGE_SIZE = 128
    NB_WORDS = 3
    LIMIT_BY_WORD = 30
    batch_size = 1
    GPU=0

    words = []

    with open("../preprocessed/words.txt", "r") as f:
        words = f.readlines()

    words = [w.strip() for w in words]

    dico = [(w, words.count(w)) for w in set(words)]
    dico.sort(key=lambda x: x[1], reverse=True)

    chosen_words = [elt[0] for elt in dico[:NB_WORDS]]


    print("Got the {} most frequent words :".format(NB_WORDS), chosen_words)

    idx = {}

    nb_images = 0
    for i, w in enumerate(words):
        if w in chosen_words:
            if w not in idx.keys():
                idx[w] = [i]
                nb_images += 1
            elif len(idx[w]) == LIMIT_BY_WORD:
                continue
            else:
                idx[w].append(i)
                nb_images += 1

    print("Writing the database")

    rmtree("../test", ignore_errors=True)
    pbar = tqdm(total=nb_images)
    for w in idx.keys():
        if not os.path.exists("../test/" + w):
            os.makedirs("../test/" + w)
        for i in idx[w]:
            pbar.update(1)
            copyfile("../preprocessed/word-{:06d}.png".format(i), "../test/" + w + "/word-{:06d}.png".format(i))
    pbar.close()

    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder("../test",
                            transform=transformation()),
        batch_size=batch_size, shuffle=True, num_workers=1
    )

    # Resnet18 is made for 
    model = models.resnet50(pretrained=True)
    model.avgpool = nn.AdaptiveAvgPool2d(1)

    if use_cuda:
        print('Using GPU')
        model.cuda(GPU)
    else:
        print('Using CPU')

    model.eval()

    outputs = []
    layers_idx = {}

    def hook(module, input, output):
        outputs.append(output)
    
    def add_hooks(model):
        for layer in model.children():
            if isinstance(layer, nn.modules.conv.Conv2d):
                layer.register_forward_hook(hook)
            else:
                add_hooks(layer)
    
    add_hooks(model)

    def get_nbchannels(model):
        liste = []
        for layer in model.children():
            if isinstance(layer, nn.modules.conv.Conv2d):
                liste.append(layer.out_channels)
            else:
                liste += get_nbchannels(layer)
        return liste

    nb_channels = get_nbchannels(model)
    nb_conv = len(nb_channels)

    rmtree("../embeddings", ignore_errors=True)
    if not os.path.exists("../embeddings"):
        os.makedirs("../embeddings")

    for i in range(nb_conv):
        if not os.path.exists("../embeddings/{:03d}".format(i)):
            os.makedirs("../embeddings/{:03d}".format(i))
    
    for i, (data, target) in tqdm(enumerate(data_loader), total=np.ceil(nb_images / batch_size)):
        model(data)
        for j, o in tqdm(enumerate(outputs), total=len(outputs)):
            if use_cuda:
                data, target = data.cuda(GPU), target.cuda(GPU)
            out = o.detach().numpy()
            out = out.reshape(out.shape[0], out.shape[1], -1)
            for k in range(out.shape[1]):
                with open("../embeddings/{:03d}/{:04d}.csv".format(j, k), "a") as f:
                    out[:,k].tofile(f, sep=",", format="%.18e")
                    f.write("\n")
        with open("../embeddings/target.csv", "a") as f:
            target.numpy().tofile(f, sep=",")
            f.write("\n")
        outputs = []
    