import os

from tqdm import tqdm

import numpy as np

from shutil import copyfile

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
        transforms.Resize((240, 600)),
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
        batch_size=1, shuffle=True, num_workers=1
    )

    if not os.path.exists("../embeddings"):
        os.makedirs("../embeddings")
    
    
    # Resnet18 is made for 
    model = models.resnet18(pretrained=True)
    model.avgpool = nn.AdaptiveAvgPool2d(1)

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

    def get_nbconv(model):
        res = 0
        for layer in model.children():
            if isinstance(layer, nn.modules.conv.Conv2d):
                res += 1
            else:
                res += get_nbconv(layer)
        return res

    nb_conv = get_nbconv(model)
    scores = []

    init = 0

    for i in tqdm(range(nb_conv)):
        count = [0 for _ in range(NB_WORDS)]
        nb_channels = 0
        a, b = 0, 0
        for j, (data, target) in tqdm(enumerate(data_loader), total=nb_images):
            model(data)
            output = outputs[i].detach().numpy()
            outputs = []
            a = output.shape[2]
            b = output.shape[3]
            output = output.reshape(-1, output.shape[1], output.shape[2] * output.shape[3])
            if j == 0:
                nb_channels = output.shape[1]
                means = np.zeros((output.shape[1], NB_WORDS, output.shape[2]))
            idx = target.numpy()
            for l, k in enumerate(list(idx)):
                means[:,k] += output[l]
                count[k] += 1
        
        for j in range(NB_WORDS):
            means[:,j] /= count[j]
        

        inner_distances = np.zeros((nb_channels, NB_WORDS))

        for j, (data, target) in tqdm(enumerate(data_loader), total=nb_images):
            model(data)
            output = outputs[i].detach().numpy()
            outputs = []
            a = output.shape[2]
            b = output.shape[3]
            output = output.reshape(-1, output.shape[1], output.shape[2] * output.shape[3])
            idx = target.numpy()
            for l,k in enumerate(list(idx)):
                inner_distances[:,k] += np.einsum("ij,ij->i", output[l] - means[:,k], output[l] - means[:,k])
        
        for j in range(NB_WORDS):
            inner_distances[:,j] /= count[j]

        outer_distances = np.zeros((nb_channels, int(NB_WORDS * (NB_WORDS - 1) / 2)))

        it = 0
        for j in range(NB_WORDS):
            for k in range(j + 1, NB_WORDS):
                outer_distances[:,it] = np.einsum("ij, ij->i", means[:,j] - means[:,k], means[:,j] - means[:,k])
                it += 1
        
        layers_idx[i] = (init, init + nb_channels)
        init += nb_channels

        scores += list(np.min(outer_distances, axis=1) / np.max(inner_distances, axis=1))
    
    plt.plot(scores)
    plt.show()

    


    
