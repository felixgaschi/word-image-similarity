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

queries = {}

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
    for data, target, indices, img_indices in tqdm(val_loader, position=0):
        if use_cuda:
            data, target = data.cuda(args.gpu), target.cuda(args.gpu)
        output = model(data)
        # sum up batch loss
        if args.estimator_type == "class":
            criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
        else:
            criterion = torch.nn.MSELoss(reduction="elementwise_mean")
            target = target.float()
            output = output[:,0]
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        if args.estimator_type == "class":
            score = output.data[:,1]
            pred = output.data.max(1, keepdim=True)[1]
        else:
            score = output.data
            pred = output.data.round()
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        for j in range(indices.size(0)):
            query = int(img_indices[j, 0])
            if query not in queries:
                queries[query] = []
            queries[query].append((
                int(img_indices[j, 1]),
                score[j].cpu().item(), 
                target[j].cpu().item()
            ))

            if target[j].cpu().item() == 1:
                nb_true += 1
                if pred[j].cpu().item() == 0:
                    nb_false_false += 1
                else:
                    nb_true_true += 1
            else:
                nb_false += 1
                if pred[j].cpu().item() == 0:
                    nb_true_false += 1
                else:
                    nb_false_true += 1
    
    mAP = 0
    Q = 0
    for q in queries.keys():
        sorted_scores = sorted(queries[q], key=lambda x: x[1], reverse=False)
        queries[q] = sorted_scores
        p_nom = 0
        p_div = 0
        cum_sum = 0
        nb_rel = 0
        for s in sorted_scores:
            if s[1] >= 0.5:
                p_div += 1
                if s[2] == 1:
                    p_nom += 1
            if s[2] == 1:
                cum_sum += p_nom * 1. / max(1, p_div)
                nb_rel += 1
        score = cum_sum * 1. / max(1, nb_rel)
        if nb_rel > 0:
            mAP += score
            Q += 1
    mAP /= max(1, Q)
    
    true_precision = nb_true_true * 1. / (nb_true_true + nb_false_true) if nb_true_true > 0 else 0.
    false_precision = nb_true_false * 1. / (nb_true_false + nb_false_false) if nb_true_false > 0 else 0.

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), mAP: {:.2f}%\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset),
        100. * mAP
    ))

    print('True positive / positive: {:.4f}'.format(nb_true_true * 1. / nb_true))
    print('True negative / negative: {:.4f}'.format(nb_true_false * 1. / nb_false))

    print('Positive precision: {:.4f}'.format(true_precision))
    print('Negative precision: {:.4f}'.format(false_precision))
    
    return 100. * correct / len(val_loader.dataset), validation_loss, mAP, nb_true_true * 1. / nb_true, nb_true_false * 1. / nb_false, true_precision, false_precision, queries
