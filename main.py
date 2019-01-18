import argparse
import os, sys
import torch
import torch.optim as optim
from torchvision import datasets
from time import time
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Word Image similarity training script')
    parser.add_argument('--data', type=str, default='../dataset', metavar='D',
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

    parser.add_argument('--save', dest="save", action="store_true")
    parser.add_argument('--no-save', dest="save", action="store_false")
    parser.set_defaults(save=False)

    parser.add_argument("--save-queries", dest="save_queries", action="store_true")
    parser.add_argument('--no-save-queries', dest="save_queries", action="store_false")
    parser.set_defaults(save_queries=False)

    parser.add_argument('--nb-workers', type=int, default=1)
    parser.add_argument('--estimator-type', type=str, default="class")
    parser.add_argument('--model', type=str, default="2channels")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--optimizer', type=str, default="SGD")
    parser.add_argument('--nb-train', type=int, default=None)
    parser.add_argument('--nb-eval', type=int, default=None)
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--nb-more', type=int, default=0)

    parser.add_argument('--eval-type', type=str, default="whole",
                        help="type of evaluation set, [toy, custom, whole]")
    parser.add_argument('--train-type', type=str, default="whole",
                        help="type of train set, [toy, custom, whole]")

    parser.add_argument('--preselect-false', dest="preselect_false", action="store_true")
    parser.add_argument('--no-preselect-false', dest="preselect_false", action="store_false")
    parser.set_defaults(preselect_false=False)

    parser.add_argument('--keep-identical', dest="keep_identical", action="store_true")
    parser.add_argument('--no-keep-identical', dest="keep_identical", action="store_false")
    parser.set_defaults(keep_identical=True)

    parser.add_argument('--augment', dest="augment", action="store_true")
    parser.add_argument('--no-augment', dest="augment", action="store_false")
    parser.set_defaults(augment=True)

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

    if args.train_type == "toy":
        train_set = data.ToyDataset(
            args.data,
            begin=1,
            end=3687,
            transform_before=data.train_transform_before if args.augment else data.train_transform_before_noaugment,
            transform_after=data.train_transform_after,
            transform_true_before=data.train_true_before,
            transform_true_after=data.train_true_after,
            more_true=args.nb_more,
            limit=args.nb_train,
            keep_identical=args.keep_identical
        )
    elif args.train_type == "custom":
        train_set = data.CustomDataset(
            args.data,
            begin=1,
            end=3687,
            transform_before=data.train_transform_before if args.augment else data.train_transform_before_noaugment,
            transform_after=data.train_transform_after,
            transform_true_before=data.train_true_before,
            transform_true_after=data.train_true_after,
            more_true=args.nb_more,
            limit=args.nb_train,
            preselect_false=args.preselect_false,
            keep_identical=args.keep_identical
        )
    else:
        train_set = data.SplitPageDataset(
            args.data,
            begin=1,
            end=3687,
            transform_before=data.train_transform_before if args.augment else data.train_transform_before_noaugment,
            transform_after=data.train_transform_after,
            transform_true_before=data.train_true_before,
            transform_true_after=data.train_true_after,
            more_true=args.nb_more,
            limit=args.nb_train,
            keep_identical=args.keep_identical
        )

    if args.eval_type == "toy":
        test_set = data.ToyDataset(
            args.data,
            begin=3687,
            end=4860,
            transform_before=data.validation_transform_before,
            transform_after=data.validation_transform_after,
            transform_true_before=None,
            transform_true_after=None,
            more_true=0,
            limit=args.nb_eval,
            preselect_false = args.preselect_false,
            keep_identical=args.keep_identical
        )
    elif args.eval_type == "custom":
        test_set = data.CustomDataset(
            args.data,
            begin=3687,
            end=4860,
            transform_before=data.validation_transform_before,
            transform_after=data.validation_transform_after,
            transform_true_before=None,
            transform_true_after=None,
            more_true=0,
            limit=args.nb_eval,
            keep_identical=args.keep_identical
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
            limit=args.nb_eval,
            keep_identical=args.keep_identical
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
        for batch_idx, (data, target, indices, _) in tqdm(enumerate(train_loader), total=len(train_loader), position=0, desc="training epoch : "+str(epoch)):
            if start_batch + batch_idx % 50000 == 0:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = args.lr * (0.1 ** ((start_batch + batch_idx) // 50000))

            if use_cuda:
                data, target = data.cuda(args.gpu), target.cuda(args.gpu)
            optimizer.zero_grad()
            output = model(data)
            if args.estimator_type == "class":
                criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
            else:
                criterion = torch.nn.MSELoss(reduction="elementwise_mean")
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

def validation(model):
    retrieved = {}
    relevantAndRetrieved = {}
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
        
        true_mAP = 0
        Q = 0
        for q in queries.keys():
            sorted_scores = sorted(queries[q], key=lambda x: x[1], reverse=False)
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
                true_mAP += score
                Q += 1
        true_mAP /= max(1, Q)
        
        scores = [relevantAndRetrieved[i] * 1. / retrieved[i] if retrieved[i] > 0 else 0. for i in retrieved.keys()]
        mAP = np.sum(scores) / len(scores) 

        true_precision = nb_true_true * 1. / (nb_true_true + nb_false_true) if nb_true_true > 0 else 0.
        false_precision = nb_true_false * 1. / (nb_true_false + nb_false_false) if nb_true_false > 0 else 0.

        mAP2 = (nb_true * true_precision + nb_false * false_precision) / (nb_true + nb_false)

        validation_loss /= len(val_loader.dataset)
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), mAP: {:.2f}%), new mAP: {:.2f}%\n'.format(
            validation_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset),
            100. * mAP,
            100. * true_mAP
        ))

        print('True positive / positive: {:.4f}'.format(nb_true_true * 1. / nb_true))
        print('True negative / negative: {:.4f}'.format(nb_true_false * 1. / nb_false))

        print('Positive precision: {:.4f}'.format(true_precision))
        print('Negative precision: {:.4f}'.format(false_precision))
    
        print("mAP over pair classes: {:.4f}".format(mAP2))
        
        return 100. * correct / len(val_loader.dataset), validation_loss, mAP, nb_true_true * 1. / nb_true, nb_true_false * 1. / nb_false, true_precision, false_precision, mAP2, queries

if __name__ == "__main__":

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
            f.write("train_acc, val_acc, val_loss, time, mAP, acc_true, acc_false, true_P, false_P, mAP2\n")

    for epoch in range(1, args.epochs + 1):
        t = time()
        train_score = train(epoch)
        test_score, loss, mAP, acc_true, acc_false, true_P, false_P, mAP2, queries = validation(model)
        if args.save_queries:
            if args.save:
                path = os.path.join(args.experiment, dirName, "queries_{:d}.txt".format(epoch))
            else:
                path = os.path.join(args.experiment, "queries_{:d}.txt".format(epoch))
            with open(path, "w") as f:
                for q in queries.keys():
                    line = ""
                    line += "{}".format(q)
                    for value in queries[q]:
                        line += ",{}".format(value)
                    line += "\n"
                    f.write(line)

        if args.save:
            model_file = os.path.join(args.experiment, dirName, 'model_' + str(epoch) + '.pth')
            torch.save(model.state_dict(), model_file)
            print('\nSaved model to ' + model_file)
        elapsed_time = time() - t
        if args.save:
            with open(os.path.join(args.experiment, dirName, "scores.csv"), "a") as f:
                f.write("{:f},{:f},{:f},{:.2f},{:f},{:f},{:f},{:f},{:f},{:f}\n".format(train_score, test_score, loss, elapsed_time, mAP, acc_true, acc_false, true_P, false_P, mAP2))
        print("Elapsed time: ", elapsed_time)
