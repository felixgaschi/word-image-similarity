from torchvision.models import resnet50
import os, argparse
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from data.transforms import MEAN, STD
import torch
import torch.nn as nn
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default="../dataset")
    parser.add_argument("--out", type=str, default="../resnet50_features")
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument('--use-gpu', dest="use_gpu", action="store_true")
    parser.add_argument('--no-use-gpu', dest='use_gpu', action="store_false")
    parser.set_defaults(use_gpu=True)   

    args = parser.parse_args()

    fnames = sorted([fn for fn in os.listdir(args.data) if fn[-4:] == ".png"])

    use_cuda = torch.cuda.is_available()
    if not args.use_gpu:
        use_cuda = False

    model = resnet50(pretrained=True)
    modules = list(model.children())[:-1]
    num_ftrs = model.fc.in_features
    model = nn.Sequential(*modules)
    for p in model.parameters():
        p.requires_grad = False
    
    if use_cuda:
        model.cuda(args.gpu)

    if not os.path.exists(args.out):
        os.makedirs(args.out)
    
    with torch.no_grad():
        model.eval()

        for i, fname in tqdm(enumerate(fnames), total=len(fnames)):

            img = Image.open(
                os.path.join(args.data, fname)
            )

            img = transforms.Resize((224, 224))(img)

            x = transforms.ToTensor()(img)
            x = transforms.Normalize(mean=[MEAN], std=[STD])(x)
            x = torch.cat((x,x,x), 0)
            x = x.unsqueeze(0)
            if use_cuda:
                x = x.cuda(args.gpu)
            x = model(x)
            x = x.view(-1)
            if use_cuda:
                x = x.cpu()
            x = x.numpy()

            np.savetxt(
                os.path.join(args.out, "word-{:06d}.png".format(i)),
                x
            )

        
