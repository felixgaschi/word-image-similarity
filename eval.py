import argparse, os
import torch
from time import time
from tqdm import tqdm
import numpy as np
from main import validation

parser = argparse.ArgumentParser(description='Word Image similarity fusion script')
parser.add_argument('--data', type=str, default='../dataset', metavar='D',
                    help="folder where data is located.")
parser.add_argument('--batch-size', type=int, default=32, metavar='B',
                    help='input batch size for training (default: 32)')
parser.add_argument('--experiment', type=str, default='../results/', metavar='E',
                    help='folder where experiment outputs are located.')
parser.add_argument('--estimator-type', type=str, default="class")
parser.add_argument('--model', type=str, default="2channels")
parser.add_argument('--model-name', type=str, default="")
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--nb-workers', type=int, default=1)
parser.add_argument('--lim', type=int, default=None)

args = parser.parse_args()

use_cuda = torch.cuda.is_available()


from models import TwoChannelsClassifier, TwoChannelsRegressor, SiameseRegressor, SiameseClassifier, PseudoSiameseClassifier, PseudoSiameseRegressor

if args.estimator_type == "class":
    if args.model == "2channels":
        model = TwoChannelsClassifier()
    elif args.model == "siamese":
        model = SiameseClassifier()
    elif args.model == "pseudosiamese":
        model = PseudoSiameseClassifier()
elif args.estimator_type == "regressor":
    if args.model == "2channels":
        model = TwoChannelsRegressor()
    elif args.model == "siamese":
        model = SiameseRegressor()
    elif args.model == "pseudosiamese":
        model = PseudoSiameseRegressor()

import data

test_set = data.SplitPageDataset(
    args.data,
    begin=3687,
    end=4860,
    transform_before=data.validation_transform_before,
    transform_after=data.validation_transform_after,
    transform_true_before=None,
    transform_true_after=None,
    more_true=0,
    keep_identical=True,
    limit=args.lim
)

val_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=args.batch_size, shuffle=True, num_workers=args.nb_workers
)

if use_cuda:
    print('Using GPU')
    model.cuda(args.gpu)
else:
    print('Using CPU')

dirName = os.path.join(args.experiment, args.model_name)
model.load_state_dict(torch.load(dirName))

validation(model)
