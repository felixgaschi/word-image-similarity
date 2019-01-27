#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 15:14:45 2019

@author: combaldieu
"""
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

parser.add_argument('--name', type=str, default=None)

parser.add_argument('--save', dest="save", action="store_true")
parser.add_argument('--no-save', dest="save", action="store_false")
parser.set_defaults(save=False)

parser.add_argument("--save-queries", dest="save_queries", action="store_true")
parser.add_argument('--no-save-queries', dest="save_queries", action="store_false")
parser.set_defaults(save_queries=False)


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


validation_set = data.ValidationDataset(
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


val_loader = torch.utils.data.DataLoader(
    validation_set,
    batch_size=args.batch_size, shuffle=False, num_workers=args.nb_workers
)

dirName1 = args.model1_name
dirName2 = args.model2_name

if use_cuda:
    print('Using GPU')
    model1.cuda(args.gpu)
    model2.cuda(args.gpu)

    model1.load_state_dict(torch.load(dirName1))
    model2.load_state_dict(torch.load(dirName2))
else:
    print('Using CPU')
    
    model1.load_state_dict(torch.load(dirName1, map_location="cpu"))
    model2.load_state_dict(torch.load(dirName2, map_location="cpu"))
    
model1.eval()
model2.eval()

def fusion():

    queries = {}

    nb_true = 0
    nb_true_true = 0
    nb_false = 0
    nb_true_false = 0
    nb_false_true = 0
    nb_false_false = 0

    with torch.no_grad():
        model1.eval()
        model2.eval()
        fusion_loss = 0
        correct = 0
        for data, target, indices, img_indices in tqdm(val_loader):
            if use_cuda:
                data, target = data.cuda(args.gpu), target.cuda(args.gpu)
            output1 = model1(data)
            output2 = model2(data)
            # sum up batch loss
            if args.estimator1_type == "class":
                criterion1 = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
                target1 = target
            else:
                criterion1 = torch.nn.MSELoss(reduction="elementwise_mean")
                target1 = target.float()
                output1 = output1[:,0]
            fusion_loss += criterion1(output1, target1).data.item()
            
            if args.estimator2_type == "class":
                criterion2 = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
                target2 = target
            else:
                criterion2 = torch.nn.MSELoss(reduction="elementwise_mean")
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
                
            score = (pred1 + pred2)/2
            pred = score.round().long()
                
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
            sorted_scores = sorted(queries[q], key=lambda x: x[1], reverse=True)
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

        fusion_loss /= len(val_loader.dataset)
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), mAP: {:.2f}%\n'.format(
            fusion_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset),
            100. * mAP
        ))
    
        return 100. * correct / len(val_loader.dataset), fusion_loss, mAP, nb_true_true * 1. / nb_true, nb_true_false * 1. / nb_false, true_precision, false_precision, queries
    

if args.save:
    if not os.path.exists(args.experiment):
        os.makedirs(args.experiment)
        dirName = "exp-00000"
    else:
        prefixes = [int(d[4:9]) for d in os.listdir(args.experiment) if d[:4] == "exp-"]
        maxi = 0 if len(prefixes) == 0 else max(prefixes)
        dirName = "exp-{:05d}".format(maxi + 1)
    if args.name is not None:
        dirName += "-" + args.name
    os.makedirs(os.path.join(args.experiment, dirName))
    with open(os.path.join(args.experiment, dirName, "info"), "w") as f:
        dict = vars(args)
        res = "\n".join(["{}: {}".format(e, dict[e]) for e in dict.keys()]) + "\n"
        f.write(res)
    with open(os.path.join(args.experiment, dirName, "scores.csv"), "w") as f:
        f.write("val_acc, val_loss, time, mAP, acc_true, acc_false, true_P, false_P\n")

    

t = time()
test_score, loss, mAP, acc_true, acc_false, true_P, false_P, queries = fusion()
if args.save_queries:
    if args.save:
        path = os.path.join(args.experiment, dirName, "queries.txt")
    else:
        path = os.path.join(args.experiment, "queries.txt")
    with open(path, "w") as f:
        for q in sorted(queries.keys()):
            line = ""
            line += "{}".format(q)
            for value in queries[q]:
                line += ",{}".format(value)
            line += "\n"
            f.write(line)

elapsed_time = time() - t
print("Elapsed time: ", elapsed_time)

if args.save:
    with open(os.path.join(args.experiment, dirName, "scores.csv"), "a") as f:
        f.write("{:f},{:f},{:.2f},{:f},{:f},{:f},{:f},{:f}\n".format(test_score, loss, elapsed_time, mAP, acc_true, acc_false, true_P, false_P))
        
