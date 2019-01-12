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


loader_controller = data.ManuallyBalancedController(
    args.data, 
    transform_eval_before=data.validation_transform_before,
    transform_eval_after=data.validation_transform_after,
    transform_before=data.train_transform_before,
    transform_after=data.train_transform_after,
    transform_true_before=None,
    nb_words_train=args.nb_train,
    nb_words_val=args.nb_eval,
    verbose=1
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
    model.cuda(args.gpu)
else:
    print('Using CPU')

if args.load :
    dirName = "/exp-{:05d}".format(args.load)
    model.load_state_dict(torch.load(args.experiment + dirName + '.pth'))
    model.eval()

if args.optimizer == "SGD" :
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0002)
elif args.optimizer == "Adam" :
    optimizer = optim.Adam(model.parameters(), lr=10**-5)

def train(epoch):
    retrieved = {}
    relevantAndRetrieved = {}

    model.train()
    correct = 0
    for batch_idx, (data, target, indices) in tqdm(enumerate(train_loader), total=len(train_loader)):
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
    
    print('\nTraining score: {}/{} ({:.0f}%; mAP: {:.2f})\n'.format(
        correct, len(train_loader.dataset), 100. * correct / len(train_loader.dataset),
        100. * mAP
    ))

    return 100. * correct / len(train_loader.dataset)

def validation():
    retrieved = {}
    relevantAndRetrieved = {}

    with torch.no_grad():
        model.eval()
        validation_loss = 0
        correct = 0
        for data, target, indices in tqdm(val_loader):
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

        validation_loss /= len(val_loader.dataset)
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), mAP: {:.2f}%\n'.format(
            validation_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset),
            100. * mAP
        ))
    
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
