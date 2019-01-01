from data.transforms import grey_pil_loader
import torch.utils.data as data
import os

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

