import torchvision.transforms as transforms
import torch.utils.data as data
import torch
from PIL import Image
import os
from tqdm import tqdm
import pickle as pk
import numpy as np

def keep2chan(x):
    return x[:2,:,:]

def train_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(np.random.randint(-10,11, size=2)),
        transforms.CenterCrop(size=(100,40)),
        transforms.Lambda(keep2chan),
        transforms.Normalize(mean=[0.45, 0.45], std=[0.22, 0.22])
    ])

def validation_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(np.random.randint(-10,11, size=2)),
        transforms.CenterCrop(size=(100,40)),
        transforms.Lambda(keep2chan),
        transforms.Normalize(mean=[0.45, 0.45], std=[0.22, 0.22])
    ])

train_transform_before = transforms.Resize((40, 100))
validation_transform_before = transforms.Resize((40, 100))        

train_transform_after = transforms.Normalize(mean=[0.45, 0.45], std=[0.22, 0.22])
validation_transform_after = transforms.Normalize(mean=[0.45, 0.45], std=[0.22, 0.22])

def grey_pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert("L")


class WordPairPickle(data.Dataset):

    def __init__(self, root, fileName, loader=grey_pil_loader, 
                transform_before=None, 
                transform_after=None,
                transform_true_before=None,
                transform_true_after=None,
                jump_equal=True, verbose=1, lim=None):
        self.root = root
        self.fileName = os.path.join(root, fileName)
        self.loader = loader
        self.transform_before = transform_before
        self.transform_after = transform_after
        self.transform_true_before = transform_true_before
        self.transform_true_after = transform_true_after

        data = []

        with open(self.fileName, "rb") as f:
            data = pk.load(f)
        
        self.data = data[:lim]

    def __getitem__(self, index):
        (a, b), target = self.data[index]
        sample1, sample2 = self.loader(a), self.loader(b)

        if self.transform_before is not None:
            sample1 = self.transform_before(sample1)
            sample2 = self.transform_before(sample2)

        if self.transform_true_before is not None and target == 1:
            sample1 = self.transform_true_before(sample1)
            sample2 = self.transform_true_before(sample2)

        sample1 = transforms.ToTensor()(sample1)
        sample2 = transforms.ToTensor()(sample2)
        sample = torch.cat((sample1, sample2), 0)

        if self.transform_after is not None:
            sample = self.transform_after(sample)

        if self.transform_true_after is not None and target == 1:
            sample = self.transform_true_after(sample)

        return (sample, target)
        
    def __len__(self):
        return len(self.data)
