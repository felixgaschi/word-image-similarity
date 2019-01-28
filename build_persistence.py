import gudhi
import math, argparse, os
from tqdm import tqdm
from PIL import Image
import numpy as np
from scipy.ndimage import distance_transform_cdt

H0_MAX = 71.02658644511894
H1_MAX = 32.63690531645403

def manhattan(img):
    dist1 = distance_transform_cdt(img, metric="taxicab")
    dist2 = distance_transform_cdt(1 - img, metric="taxicab")
    return dist1 - dist2

def w(b,d, C = 0.5, p = 1):
    return(np.arctan(C*(d-b)**p))

def get_persistence_image(img, thresh=128, C=0.5, p=1, size=51,sigma=0.5,
                         win = {'xmin' : -10.5, 'xmax' : 60.5, 'ymin' : -10.5, 'ymax' : 70.5}):
    arr = np.array(img)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] > thresh:
                arr[i, j] = 1
            else:
                arr[i, j] = 0
    
    fil = manhattan(arr)
    
    cp = gudhi.CubicalComplex(
        dimensions=fil.shape, 
        top_dimensional_cells=[fil[i,j] for i in range(fil.shape[0] - 1, -1, -1) for j in range(fil.shape[1])]
    )
    
    pers = cp.persistence()
    
    h0 = [p[1] for p in pers if p[0] == 0]
    h1 = [p[1] for p in pers if p[0] == 1]
    
    def rho(x, y, points):
        somme = 0
        for p in points:
            we = 1
            if p[1] == math.inf:
                we = 1.
                a = np.exp(-((p[0]-x)**2+(win["ymax"]-y)**2)/(2*sigma**2))
            else:
                we = w(p[0], p[1])
                a = np.exp(-((p[0]-x)**2+(p[1]-y)**2)/(2*sigma**2))
            somme += we * a
        return somme
    xx = np.linspace(win['xmin'],win['xmax'], size)
    yy = np.linspace(win['ymax'],win['ymin'], size)
    image0 = np.zeros((size, size))
    image1 = np.zeros((size, size))
    for j,x in enumerate(xx):
        for i,y in enumerate(yy):
            if (y >= x):
                image0[i,j] = rho(x,y,h0)
                image1[i,j] = rho(x,y,h1)
            
    return image0, image1

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generating matching dataset for MNIST')

    parser.add_argument("--data", type=str, default="../dataset", metavar="I",
                        help="directory for the data")
    parser.add_argument("--out", type=str, default="../persistence", metavar="O")

    parser.add_argument('--save', dest="save", action="store_true")
    parser.add_argument('--no-save', dest="save", action="store_false")
    parser.set_defaults(save=False)

    args = parser.parse_args()


    fnames = [f for f in os.listdir(args.data) if f[-4:] == ".png"]
    
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    max0 = 0
    max1 = 0

    for fname in tqdm(fnames):
        img = Image.open(os.path.join(args.data, fname))

        image0, image1 = get_persistence_image(img)

        max0 = max(max0, np.max(image0))
        max1 = max(max1, np.max(image1))

        if args.save:
            img0 = Image.fromarray(255 * image0 / H0_MAX)
            img1 = Image.fromarray(255 * image1 / H1_MAX)

            img0.convert(mode="L").save(os.path.join(args.out, fname[:-4] + "_0.png"))
            img1.convert(mode="L").save(os.path.join(args.out, fname[:-4] + "_1.png"))

    print("maximum of h0: ", max0)
    print("maximum of h1: ", max1)
