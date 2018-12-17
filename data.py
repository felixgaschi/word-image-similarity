import torchvision.transforms as transforms


def keep2chan(x):
    return x[:2,:,:]

def train_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(keep2chan),
        transforms.Normalize(mean=[0.45, 0.45], std=[0.22, 0.22])
    ])

def validation_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(keep2chan),
        transforms.Normalize(mean=[0.45, 0.45], std=[0.22, 0.22])
    ])
