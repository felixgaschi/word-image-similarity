import argparse
import os, sys
import torch
import torch.optim as optim
from torchvision import datasets
from time import time
from tqdm import tqdm
import numpy as np



parser = argparse.ArgumentParser(description='Word Image similarity training script')
parser.add_argument('--data', type=str, default='../preprocessed', metavar='D',
                    help="folder where data is located.")
parser.add_argument('--batch-size', type=int, default=32, metavar='B',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--experiment', type=str, default='../results/', metavar='E',
                    help='folder where experiment outputs are located.')
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--save', type=bool, default=False)
parser.add_argument('--nb-workers', type=int, default=1)
parser.add_argument('--estimator-type', type=str, default="class")
parser.add_argument('--model', type=str, default="siamese")
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--optimizer', type=str, default="SGD")
parser.add_argument('--nb-train', type=int, default=None)
parser.add_argument('--nb-eval', type=int, default=None)
parser.add_argument('--load', type=int, default=0)
parser.add_argument('--nb-more', type=int, default=0)
parser.add_argument('--eval-toy', type=bool, default=False)
parser.add_argument('--train-toy', type=bool, default=False)

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

if args.train_toy:
    train_set = data.ToyDataset(
        args.data,
        begin=1,
        end=3687,
        transform_before=data.train_transform_before,
        transform_after=data.train_transform_after,
        transform_true_before=data.train_true_before,
        transform_true_after=data.train_true_after,
        more_true=args.nb_more,
        limit=args.nb_train
    )
else:
    train_set = data.SplitPageDataset(
        args.data,
        begin=1,
        end=3687,
        transform_before=data.train_transform_before,
        transform_after=data.train_transform_after,
        transform_true_before=data.train_true_before,
        transform_true_after=data.train_true_after,
        more_true=args.nb_more,
        limit=args.nb_train
    )

if args.eval_toy:
    test_set = data.ToyDataset(
        args.data,
        begin=3687,
        end=4860,
        transform_before=data.validation_transform_before,
        transform_after=data.validation_transform_after,
        transform_true_before=None,
        transform_true_after=None,
        more_true=0,
        limit=args.nb_eval
    )
else:
    test_set = data.SplitPageDataset(
        args.data,
        begin=3687,
        end=4860,
        transform_before=data.validation_transform_before,
        transform_after=data.validation_transform_after,
        transform_true_before=None,
        transform_true_after=None,
        more_true=0,
        limit=args.nb_eval
    )

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=args.batch_size, shuffle=True, num_workers=args.nb_workers
)

val_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=args.batch_size, shuffle=True, num_workers=args.nb_workers
)

nb_false, nb_true, more_true = train_set.get_info()
print("Info about training set:\nnb false:{:d}\nnb true:{:d}\nadditionnal transformed true:{:d}\n".format(
    nb_false, nb_true, more_true
))

nb_false, nb_true, more_true = test_set.get_info()
print("Info about validation set:\nnb false:{:d}\nnb true:{:d}\nadditionnal transformed true:{:d}\n".format(
    nb_false, nb_true, more_true
))

if use_cuda:
    print('Using GPU')
    model.cuda(args.gpu)
else:
    print('Using CPU')

if args.load :
    dirName = "/exp-{:05d}".format(args.load)
    model.load_state_dict(torch.load(args.experiment + dirName + '.pth'))
    model.eval()

if args.optimizer == "SGD" :
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0002)
elif args.optimizer == "Adam" :
    optimizer = optim.Adam(model.parameters(), lr=10**-5)

def train(epoch):
    retrieved = {}
    relevantAndRetrieved = {}
    
    start_batch = epoch * len(train_loader)

    model.train()
    correct = 0
    for batch_idx, (data, target, indices) in tqdm(enumerate(train_loader), total=len(train_loader), position=0):
        if start_batch + batch_idx % 50000 == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr * (0.1 ** ((start_batch + batch_idx) // 50000))

        if use_cuda:
            data, target = data.cuda(args.gpu), target.cuda(args.gpu)
        optimizer.zero_grad()
        output = model(data)
        if args.estimator_type == "class":
            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        else:
            criterion = torch.nn.MSELoss(reduction="mean")
            target = target.float()
            output = output[:,0]
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if args.estimator_type == "class":
            pred = output.data.max(1, keepdim=True)[1]
        else:
            pred = output.data.round()
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        for j in range(indices.size(0)):
            ref = int(indices[j, 0])

            if pred[j].cpu().item() == 1:
                if ref not in retrieved.keys():
                    relevantAndRetrieved[ref] = 0
                    retrieved[ref] = 0
                retrieved[ref] += 1
            if pred[j].cpu().item() == 1 and target[j].cpu().item() == 1:
                relevantAndRetrieved[ref] += 1
    
    scores = [relevantAndRetrieved[i] * 1. / retrieved[i] for i in retrieved.keys() if retrieved[i] > 0]
    mAP = np.sum(scores) / len(scores)
    
    print('\nTraining score: {}/{} ({:.0f}%; mAP: {:.2f}%)\n'.format(
        correct, len(train_loader.dataset), 100. * correct / len(train_loader.dataset),
        100. * mAP
    ))

    return 100. * correct / len(train_loader.dataset)

def validation():
    retrieved = {}
    relevantAndRetrieved = {}

    nb_true = 0
    nb_true_true = 0
    nb_false = 0
    nb_true_false = 0

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
                else:
                    nb_false += 1
                    if pred[j].cpu().item() == 0:
                        nb_true_false += 1

        scores_1 = [relevantAndRetrieved[i] * 1. / retrieved[i] if retrieved[i] > 0 else 0. for i in retrieved.keys() if test_set.words[i] in train_set.word_set]
        scores_2 = [relevantAndRetrieved[i] * 1. / retrieved[i] if retrieved[i] > 0 else 0. for i in retrieved.keys() if test_set.words[i] not in train_set.word_set]
        scores = scores_1 + scores_2
        mAP = np.sum(scores) / len(scores) 
        mAP_1 = np.sum(scores_1) / len(scores_1)
        mAP_2 = np.sum(scores_2) / len(scores_2)

        validation_loss /= len(val_loader.dataset)
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), mAP: {:.2f}% (same: {:.2f}% ; difft: {:.2f}%)\n'.format(
            validation_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset),
            100. * mAP, 100. * mAP_1, 100. * mAP_2
        ))

        print('True positive / positive: {:.4f}'.format(nb_true_true * 1. / nb_true))
        print('True negative / negative: {:.4f}'.format(nb_true_false * 1. / nb_false))
    
        return 100. * correct / len(val_loader.dataset)

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

for epoch in range(1, args.epochs + 1):
    t = time()
    train_score = train(epoch)
    test_score = validation()
    if args.save:
        model_file = os.path.join(args.experiment, dirName, 'model_' + str(epoch) + '.pth')
        torch.save(model.state_dict(), model_file)
        print('\nSaved model to ' + model_file)
    elapsed_time = time() - t
    if args.save:
        with open(os.path.join(args.experiment, dirName, "scores.csv"), "a") as f:
            f.write("{:f},{:f},{:.2f}\n".format(train_score, test_score, elapsed_time))
    print("Elapsed time: ", elapsed_time)
