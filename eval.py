import argparse, os
import torch
from time import time
from tqdm import tqdm
import numpy as np

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

retrieved = {}
relevantAndRetrieved = {}

nb_true = 0
nb_true_true = 0
nb_false = 0
nb_true_false = 0
nb_false_true = 0
nb_false_false = 0

with torch.no_grad():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target, indices in tqdm(val_loader, position=0):
        if use_cuda:
            data, target = data.cuda(args.gpu), target.cuda(args.gpu)
        output = model(data)
        # sum up batch loss
        if args.estimator_type == "class":
            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        else:
            criterion = torch.nn.MSELoss(reduction="mean")
            target = target.float()
            output = output[:,0]
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        if args.estimator_type == "class":
            pred = output.data.max(1, keepdim=True)[1]
        else:
            pred = output.data.round()
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        for j in range(indices.size(0)):
            ref = int(indices[j, 0])
            other = int(indices[j, 1])

            if pred[j].cpu().item() == 1:
                if ref not in retrieved.keys():
                    relevantAndRetrieved[ref] = 0
                    retrieved[ref] = 0
                retrieved[ref] += 1
                if target[j].cpu().item() == 1:
                    relevantAndRetrieved[ref] += 1
                    nb_true_true += 1
            if target[j].cpu().item() == 1:
                nb_true += 1
                if pred[j].cpu().item() == 0:
                    nb_false_false += 1
            else:
                nb_false += 1
                if pred[j].cpu().item() == 0:
                    nb_true_false += 1
                else:
                    nb_false_true += 1

    scores = [relevantAndRetrieved[i] * 1. / retrieved[i] if retrieved[i] > 0 else 0. for i in retrieved.keys()]
    mAP = np.sum(scores) / len(scores) 

    true_precision = nb_true_true * 1. / (nb_true_true + nb_false_true)
    false_precision = nb_true_false * 1. / (nb_true_false + nb_false_false)

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), mAP: {:.2f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset),
        100. * mAP
    ))

    print('True positive / positive: {:.4f}'.format(nb_true_true * 1. / nb_true))
    print('True negative / negative: {:.4f}'.format(nb_true_false * 1. / nb_false))

    print('Positive precision: {:.4f}'.format(true_precision))
    print('Negative precision: {:.4f}'.format(false_precision))
