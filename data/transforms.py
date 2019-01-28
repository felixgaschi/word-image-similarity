import torchvision.transforms as transforms 
import torch.utils.data as data 
import torch 
from PIL import Image 
import os 
from tqdm import tqdm 
import pickle as pk 
import numpy as np 
import re

BINARY_MEAN = 0.91
BINARY_STD = 0.24
MEAN = 0.77
STD = 0.17
H0_MEAN = 0.0012
H0_STD = 0.31
H1_MEAN = 0.0023
H1_STD = 0.58

def normalize_string(word):
    return re.sub("[^\w\s]", "", word).lower()


def longestSubstringFinder(string1, string2):
    if string1 == string2:
        return string1
    answer = ""
    len1, len2 = len(string1), len(string2)
    for i in range(len1):
        match = ""
        for j in range(len2):
            if (i + j < len1 and string1[i + j] == string2[j]):
                match += string2[j]
            else:
                if (len(match) > len(answer)): answer = match
                match = ""
    return answer

def get_score(w1, w2, type="equal"):
    if type == "equal":
        return 1 if w1 == w2 else 0
    else:
        return len(longestSubstringFinder(w1, w2)) * 1. / max(1, len(w1), len(w2))


string_transform = {
    "strict": lambda w: w,
    "lower": lambda w: w.lower(),
    "ponctuation": lambda w: re.sub("[^\w\s]", "", w),
    "all": lambda w: re.sub("[^\w\s]", "", w).lower()
}


def jitter(img, S=(5,5)):
    a = np.array(img)
    x = np.random.randint(0, 2 * S[0] + 1)
    y = np.random.randint(0, 2 * S[1] + 1)
    return Image.fromarray(a[x:a.shape[0] - 2 * S[0] + x,y:a.shape[1] - 2 * S[1] + y])

def equalize(img, args):
    hist = np.loadtxt(os.path.join(args.data, "history.txt"))
    arr = np.array(img)
    arr2 = np.zeros(arr.shape)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            index = int(arr[i, j])
            arr2[i, j] = 255 * np.sum(hist[:index + 1])
    return Image.fromarray(arr2)

def train_transform_false_before(args):
    trans = []
    if args.equalize:
        trans.append(
            transforms.Lambda(lambda x: equalize(x, args))
        )
    if args.shearing > 0.:
        trans.append(
            transforms.RandomAffine(0,shear=(-args.shearing, args.shearing), fillcolor=210)
        )
    if args.augment_false:
        trans.append(
            transforms.Lambda(lambda x: jitter(x))
        )
    if args.model == "resnet50":
        trans += [
            transforms.Resize((224, 224))
        ]
    else:
        trans += [
            transforms.Resize((40, 100))
        ]
    return transforms.Compose(trans)

def transform_after(args):
    trans = []
    if args.binarize:
        trans += [
            transforms.Lambda(lambda x: x > 0.5),
            transforms.Lambda(lambda x: x.float()),
        ]
        if args.normalize:
            trans.append(
                transforms.Normalize(mean=[BINARY_MEAN] * 2, std=[BINARY_STD] * 2)
            )
    elif args.normalize:
        trans += [
            transforms.Normalize(mean=[MEAN] * 2, std=[STD] * 2)
        ]
    return transforms.Compose(trans)

def train_transform_true_before(args):
    trans = []
    if args.shearing > 0.:
        trans.append(
            transforms.RandomAffine(0,shear=(-args.shearing, args.shearing), fillcolor=210)
        )
    if args.augment:
        trans.append(
            transforms.Lambda(lambda x: jitter(x))
        )
    if args.model == "resnet50":
        trans += [
            transforms.Resize((224, 224))
        ]
    else:
        trans += [
            transforms.Resize((40, 100))
        ]
    return transforms.Compose(trans)


def validation_transform_before(args):
    if args.model == "resnet50":
        return transforms.Resize((224, 224))
    else:
        return transforms.Resize((40, 100))
        
def noise(img, std=5):
    return img + torch.randn(img.size())*std

def grey_pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert("L")

class SplitPageDataset(data.Dataset):

    def __init__(self, root, begin=1, end=3697,
                loader=None,
                more_true=0,
                limit=None,
                keep_identical=False,
                matching="strict",
                score="equal",
                ):
        self.root = root
        self.begin = begin
        self.end = end
        self.loader = loader
        self.more_true = more_true
        self.keep_identical = keep_identical
        self.transform = string_transform[matching]
        self.score = score

        words = []
        indices = {}

        with open(os.path.join(root, "words.txt"), "r") as f:
            for i, line in enumerate(f):
                w = self.transform(line.strip())
                words.append(w)

        if end is None:
            end = len(words)
        self.end = end

        with open(os.path.join(root, "words.txt"), "r") as f:
            for i, line in enumerate(f):
                w = self.transform(line.strip())
                if i >= self.begin and i < self.end:
                    if w not in indices.keys():
                        indices[w] = [i]
                    else:
                        indices[w].append(i)

        self.words = words
        self.indices = indices
        self.word_set = set(words)
        self.id2word = {i:w for i,w in enumerate(self.word_set)}
        self.word2id = {w:i for i,w in enumerate(self.word_set)}

        length = (end - begin) ** 2 if keep_identical else (end - begin) * (end - begin - 1)
        if limit is not None:
            self.limit = min(length, limit)
            self.end = int(np.ceil(np.sqrt(self.limit))) + begin
        else:
            self.limit = length
            self.end = self.end
        
        self.true_indices = self.get_indices_true()
        assert len(self.true_indices) > 0 or self.more_true == 0, "The dataset is too small to contain any true pair. " \
                                                                  + "You should use ToyDataset instead or choose " \
                                                                  + "a larger number of samples."
        self.length = self.limit + self.more_true
    
    def get_file(self, id):
        return os.path.join(self.root, "word-{:06d}.png".format(id))

    def get_indices_target(self, index):
        if index < self.limit:
            indexA = self.begin + index // (self.end - self.begin)
            if not self.keep_identical:
                indexB = self.begin + index % (self.end - self.begin - 1)
                if indexB >= indexA:
                    indexB += 1
            else:
                indexB = self.begin + index % (self.end - self.begin)
            w1, w2 = self.words[indexA], self.words[indexB]
            target = get_score(w1, w2, type=self.score)
            idA, idB = self.word2id[w1], self.word2id[w2]
        else:
            index -= self.limit
            newIndex = np.random.choice(self.true_indices)
            return self.get_indices_target(newIndex)
        return indexA, indexB, target, idA, idB

    def __getitem__(self, index):
        indexA, indexB, target, idA, idB = self.get_indices_target(index)
        
        indices = torch.tensor([idA, idB], dtype=torch.int)
        img_indices = torch.tensor([indexA, indexB], dtype=torch.int)

        sample = self.loader.get_sample(indexA, indexB, target)

        return (sample, target, indices, img_indices)
    
    def __len__(self):
        return self.length

    def get_info(self):
        nb_false = 0
        nb_true = 0
        for i in range(self.limit):
            _, _, target, _, _ = self.get_indices_target(i)
            if target == 1:
                nb_true += 1
            else:
                nb_false += 1
        return nb_false, nb_true, self.more_true
    
    def get_indices_true(self):
        res = []
        for i in range(self.limit):
            _, _, target, _, _ = self.get_indices_target(i)
            if target == 1:
                res.append(i)
        return res


class CustomDataset(data.Dataset):

    def __init__(self, root, begin=1, end=3697,
                loader=None,
                more_true=0,
                limit=None,
                keep_identical=False,
                preselect_false=False,
                remove_hard=False,
                matching="strict",
                score="equal"):

        self.root = root
        self.begin = begin
        self.end = end
        self.loader = loader
        self.more_true = more_true
        self.keep_identical = keep_identical
        self.limit = limit
        self.preselect_false = preselect_false
        self.remove_hard = remove_hard
        self.transform = string_transform[matching]
        self.score = score

        words = []
        indices = {}

        with open(os.path.join(root, "words.txt"), "r") as f:
            for i, line in enumerate(f):
                w = self.transform(line.strip())
                words.append(w)

        if end is None:
            end = len(words)
        self.end = end
        
        with open(os.path.join(root, "words.txt"), "r") as f:
            for i, line in enumerate(f):
                w = self.transform(line.strip())
                if i >= self.begin and i < self.end:
                    if w not in indices.keys():
                        indices[w] = [i]
                    else:
                        indices[w].append(i)

        self.words = words
        self.indices = indices
        self.word_set = set(words)
        self.id2word = {i:w for i,w in enumerate(self.word_set)}
        self.word2id = {w:i for i,w in enumerate(self.word_set)}

        self.true_pairs_id = []
        print("Building true pairs list")
        for ids in tqdm(self.indices.values()):
            for i in ids:
                for j in ids:
                    if j == i and not keep_identical:
                        continue
                    if self.limit is not None and self.limit <= len(self.true_pairs_id):
                        break
                    self.true_pairs_id.append((i, j))
                if self.limit is not None and self.limit <= len(self.true_pairs_id):
                    break
            if self.limit is not None and self.limit <= len(self.true_pairs_id):
                break

        self.nb_true = len(self.true_pairs_id) + self.more_true

        if preselect_false:
            self.false_pairs_id = []
            print("Building false pairs list")
            pbar2 = tqdm(total=self.nb_true)
            while len(self.false_pairs_id) < self.nb_true:
                i, j = np.random.choice(range(self.begin, self.end), replace=False, size=(2,))
                if self.words[i] != self.words[j] and (i, j) not in self.false_pairs_id:
                    if remove_hard and normalize_string(self.words[i]) == normalize_string(self.words[j]):
                        continue
                    self.false_pairs_id.append((i, j))
                    pbar2.update(1)
            pbar2.close()

        self.length = 2 * self.nb_true

    def get_file(self, id):
        return os.path.join(self.root, "word-{:06d}.png".format(id))

    def __getitem__(self, index):
        if index % 2 == 0:
            indexA, indexB = self.true_pairs_id[(index // 2) % len(self.true_pairs_id)]
            target = get_score(self.words[indexA], self.words[indexB], type=self.score)
        else:
            if self.preselect_false:
                indexA, indexB = self.false_pairs_id[index // 2]
            else:
                indexA, indexB = np.random.choice(range(self.begin, self.end), replace=False, size=(2,))
                while self.words[indexA] == self.words[indexB] or \
                        (self.remove_hard and normalize_string(self.words[indexA]) == normalize_string(self.words[indexB])):
                    indexA, indexB = np.random.choice(range(self.begin, self.end), replace=False, size=(2,))
            target = get_score(self.words[indexA], self.words[indexB], type=self.score)
        w1, w2 = self.words[indexA], self.words[indexB]
        idA, idB = self.word2id[w1], self.word2id[w2]

        word_indices = torch.tensor([idA, idB], dtype=torch.int)
        img_indices = torch.tensor([indexA, indexB], dtype=torch.int)

        sample = self.loader.get_sample(indexA, indexB, target)

        return (sample, target, word_indices, img_indices)

    def __len__(self):
        return self.length

    def get_info(self):
        return len(self.true_pairs_id) + self.more_true, len(self.true_pairs_id), self.more_true


class ValidationDataset(SplitPageDataset):

    def __init__(self, *args, **kwargs):
        self.queries = []

        super(ValidationDataset, self).__init__(*args, **kwargs)

        for w in self.indices.keys():
            if len(self.indices[w]) > 1:
                self.queries += self.indices[w]
        
        if "limit" in kwargs.keys() and kwargs["limit"] is not None:
            self.length = kwargs["limit"]
        else:
            if self.keep_identical:
                self.length = len(self.queries) * (self.end - self.begin)
            else:
                self.length = len(self.queries) * (self.end - self.begin - 1)
            self.limit = self.length
    

    def get_indices_target(self, index):
        if self.queries == []:
            return super(ValidationDataset, self).get_indices_target(index)
        indexA = self.queries[index // (self.end - self.begin)]
        if not self.keep_identical:
            indexB = self.begin + index % (self.end - self.begin - 1)
            if indexB >= indexA:
                indexB += 1
        else:
            indexB = self.begin + index % (self.end - self.begin)
        w1, w2 = self.words[indexA], self.words[indexB]
        target = get_score(w1, w2, type=self.score)
        idA, idB = self.word2id[w1], self.word2id[w2]
        return indexA, indexB, target, idA, idB


class Loader:

    def __init__(self, root, loader=grey_pil_loader):
        self.root = root
        self.loader = grey_pil_loader

    def get_file(self, id):
        return os.path.join(self.root, "word-{:06d}.png".format(id))
    
    def get_sample(self, indexA, indexB, target):
        raise NotImplementedError("Loader is an abstract class")


class ImagePairLoader(Loader):

    def __init__(self, root, loader=grey_pil_loader, transform_false_before=None, transform_true_before=None, \
    transform_false_after=None, transform_true_after=None):
        super(ImagePairLoader, self).__init__(root, loader=loader)

        self.transform_false_before = transform_false_before
        self.transform_true_before = transform_true_before
        self.transform_false_after = transform_false_after
        self.transform_true_after = transform_true_after

    def get_sample(self, indexA, indexB, target):
        fname_1, fname_2 = self.get_file(indexA), self.get_file(indexB)
        
        sample1, sample2 = self.loader(fname_1), self.loader(fname_2)

        if self.transform_false_before is not None and target <= 0.5:
            sample1 = self.transform_false_before(sample1)
            sample2 = self.transform_false_before(sample2)

        if self.transform_true_before is not None and target >= 0.5:
            sample1 = self.transform_true_before(sample1)
            sample2 = self.transform_true_before(sample2)

        sample1 = transforms.ToTensor()(sample1)
        sample2 = transforms.ToTensor()(sample2)
        sample = torch.cat((sample1, sample2), 0)

        if self.transform_false_after is not None and target <= 0.5:
            sample = self.transform_false_after(sample)

        if self.transform_true_after is not None and target >= 0.5:
            sample = self.transform_true_after(sample)
        
        return sample


class FeatureLoader(Loader):

    def get_sample(self, indexA, indexB, target):
        
        fname_i, fname_j = self.get_file(indexA), self.get_file(indexB)
        x1, x2 = np.loadtxt(fname_i), np.loadtxt(fname_j)

        x1, x2 = torch.from_numpy(x1), torch.from_numpy(x2)
        sample = torch.cat((x1, x2), 0)
        sample = sample.float()

        return sample


class PersistenceLoader(Loader):

    def __init__(self, root, persistence_root, loader=grey_pil_loader, transform_false_before=None, transform_true_before=None, \
    transform_false_after=None, transform_true_after=None):
        super(PersistenceLoader, self).__init__(root, loader=grey_pil_loader)

        self.image_loader = ImagePairLoader(
            root, 
            loader=grey_pil_loader, 
            transform_false_before=transform_false_before,
            transform_true_before=transform_true_before,
            transform_false_after=transform_false_after,
            transform_true_after=transform_true_after
        )

        self.persistence_root = persistence_root
        

    def get_file(self, id):
        return os.path.join(self.root, "word-{:06d}.png".format(id)), \
            os.path.join(self.persistence_root, "word-{:06d}_0.png".format(id)), \
            os.path.join(self.persistence_root, "word-{:06d}_1.png".format(id))
    
    def get_sample(self, indexA, indexB, target):
        fnameA, fnameA_0, fnameA_1 = self.get_file(indexA)
        fnameB, fnameB_0, fnameB_1 = self.get_file(indexB)

        sample = self.image_loader.get_sample(indexA, indexB, target)

        hA_0, hA_1 = self.loader(fnameA_0), self.loader(fnameA_1)
        hB_0, hB_1 = self.loader(fnameB_0), self.loader(fnameB_1)

        liste = [hA_0, hA_1, hB_0, hB_1]

        for i in range(len(liste)):
            tmp = liste[i]
            tmp = transforms.Resize((40, 100))(tmp)
            tmp = transforms.ToTensor()(tmp)
            if i % 2 == 0:
                tmp = transforms.Normalize(mean=[H0_MEAN], std=[H0_STD])(tmp)
            else:
                tmp = transforms.Normalize(mean=[H1_MEAN], std=[H1_STD])(tmp)
            liste[i] = tmp
        
        sample = torch.cat((sample[:1,], liste[0], liste[1], sample[1:,], liste[2], liste[3]), 0)

        return sample
