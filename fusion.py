#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 15:14:45 2019

@author: combaldieu
"""
import argparse
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
parser.add_argument('--estimator1-type', type=str, default="class")
parser.add_argument('--model1', type=str, default="2channels")
parser.add_argument('--model1-name', type=str, default="")
parser.add_argument('--estimator2-type', type=str, default="regressor")
parser.add_argument('--model2', type=str, default="2channels")
parser.add_argument('--model2-name', type=str, default="")
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--split', type=int, default=3697)

parser.add_argument('--binarize', dest="binarize", action="store_true")
parser.add_argument('--no-binarize', dest="binarize", action="store_false")
parser.set_defaults(binarize=False)

parser.add_argument('--equalize', dest="equalize", action="store_true")
parser.add_argument('--no-equalize', dest="equalize", action="store_false")
parser.set_defaults(equalize=False)

parser.add_argument('--normalize', dest="normalize", action="store_true")
parser.add_argument('--no-normalize', dest="normalize", action="store_false")
parser.set_defaults(normalize=True)

parser.add_argument('--nb-eval', type=int, default=None)

parser.add_argument('--keep-identical', dest="keep_identical", action="store_true")
parser.add_argument('--no-keep-identical', dest="keep_identical", action="store_false")
parser.set_defaults(keep_identical=False)

parser.add_argument('--matching', type=str, default="strict",
                        help="[strict, lower, ponctuation, all]")

parser.add_argument('--nb-workers', type=int, default=1)


args = parser.parse_args()

use_cuda = torch.cuda.is_available()


from models import TwoChannelsClassifier, TwoChannelsRegressor, SiameseRegressor, SiameseClassifier, PseudoSiameseClassifier, PseudoSiameseRegressor

if args.estimator1_type == "class":
    if args.model1 == "2channels":
        model1 = TwoChannelsClassifier()
    elif args.model1 == "siamese":
        model1 = SiameseClassifier()
    elif args.model1 == "pseudosiamese":
        model1 = PseudoSiameseClassifier()
elif args.estimator1_type == "regressor":
    if args.model1 == "2channels":
        model1 = TwoChannelsRegressor()
    elif args.model1 == "siamese":
        model1 = SiameseRegressor()
    elif args.model1 == "pseudosiamese":
        model1 = PseudoSiameseRegressor()
        
if args.estimator2_type == "class":
    if args.model2 == "2channels":
        model2 = TwoChannelsClassifier()
    elif args.model2 == "siamese":
        model2 = SiameseClassifier()
    elif args.model2 == "pseudosiamese":
        model2 = PseudoSiameseClassifier()
elif args.estimator2_type == "regressor":
    if args.model2 == "2channels":
        model2 = TwoChannelsRegressor()
    elif args.model2 == "siamese":
        model2 = SiameseRegressor()
    elif args.model2 == "pseudosiamese":
        model2 = PseudoSiameseRegressor()

import data


loader_controller = data.ValidationDataset(
    args.data,
    begin=args.split,
    end=None,
    transform_false_before=data.validation_transform_before(args),
    transform_false_after=data.transform_after(args),
    transform_true_before=data.validation_transform_before(args),
    transform_true_after=data.transform_after(args),
    more_true=0,
    limit=args.nb_eval,
    keep_identical=args.keep_identical,
    matching=args.matching
)

train_loader = torch.utils.data.DataLoader(
    loader_controller.training_set,
    batch_size=args.batch_size, shuffle=True, num_workers=args.nb_workers
)

val_loader = torch.utils.data.DataLoader(
    loader_controller.evaluation_set,
    batch_size=args.batch_size, shuffle=False, num_workers=args.nb_workers
)

if use_cuda:
    print('Using GPU')
    model1.cuda(args.gpu)
    model2.cuda(args.gpu)
else:
    print('Using CPU')
    
dirName1 = args.model1_name
model1.load_state_dict(torch.load(dirName1))
model1.eval()

dirName2 = args.model2_name
model2.load_state_dict(torch.load(dirName2))
model2.eval()

def fusion():
    retrieved = {}
    relevantAndRetrieved = {}

    with torch.no_grad():
        model1.eval()
        model2.eval()
        fusion_loss = 0
        correct = 0
        for data, target, indices in tqdm(val_loader):
            if use_cuda:
                data, target = data.cuda(args.gpu), target.cuda(args.gpu)
            output1 = model1(data)
            output2 = model2(data)
            # sum up batch loss
            if args.estimator1_type == "class":
                criterion1 = torch.nn.CrossEntropyLoss(reduction='mean')
            else:
                criterion1 = torch.nn.MSELoss(reduction="mean")
                target1 = target.float()
                output1 = output1[:,0]
            fusion_loss += criterion1(output1, target1).data.item()
            
            if args.estimator2_type == "class":
                criterion2 = torch.nn.CrossEntropyLoss(reduction='mean')
            else:
                criterion2 = torch.nn.MSELoss(reduction="mean")
                target2 = target.float()
                output2 = output2[:,0]
            fusion_loss += criterion2(output2, target2).data.item()
            
            # get the index of the max log-probability
            if args.estimator2_type == "class":
                #pred2 = output2.data.max(1, keepdim=True)[1]
                pred2 = output2.data[:,1]
            else:
                #pred2 = output2.data.round()
                pred2 = output2.data
                
            if args.estimator1_type == "class":
                #pred1 = output1.data.max(1, keepdim=True)[1]
                pred1 = output1.data[:,1]
            else:
                #pred1 = output1.data.round()
                pred1 = output1.data
                
            pred = (pred1 + pred2)/2
            pred = pred.round()
                
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            for j in range(indices.size(0)):
                ref = int(indices[j, 0])
                if ref not in retrieved.keys():
                    retrieved[ref] = 0
                if ref not in relevantAndRetrieved.keys():
                    relevantAndRetrieved[ref] = 0
                other = int(indices[j, 1])

                if pred[j].cpu().item() == 1:
                    retrieved[ref] += 1
                if pred[j].cpu().item() == 1 and target[j].cpu().item() == 1:
                    relevantAndRetrieved[ref] += 1

        scores = [relevantAndRetrieved[i] * 1. / retrieved[i] for i in retrieved.keys() if retrieved[i] > 0]
        mAP = np.sum(scores) / len(scores)

        fusion_loss /= len(val_loader.dataset)
        print('\nFusion set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), mAP: {:.2f}%\n'.format(
            fusion_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset),
            100. * mAP
        ))
    
        return 100. * correct / len(val_loader.dataset)
    
    
for _ in range(10):
    t = time()
    fusion_score = fusion()
    elapsed_time = time() - t
    print("Elapsed time: ", elapsed_time)
