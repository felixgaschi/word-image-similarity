import os

from tqdm import tqdm

import numpy as np
import pandas as pd

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
    labels = np.loadtxt("../embeddings/target.csv", dtype=int)

    
    