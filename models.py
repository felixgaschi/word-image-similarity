import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class TwoChannelsClassifier(nn.Module):
    """
    Implementation of the two-channels classifier network proposed in Zhong et al. (2018)
    https://hal.archives-ouvertes.fr/hal-01374401/document
    """

    def __init__(self):
        super(TwoChannelsClassifier, self).__init__()

        self.conv1 = nn.Conv2d(2, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 96, kernel_size=3)

        self.fc1 = nn.Linear(1344, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)
    

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2, stride=2))
        x = F.relu(F.max_pool2d(self.conv3(x), 3, stride=3))
        x = x.view(-1, 1344)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x
    
class TwoChannelsRegressor(nn.Module):
    """
    Implementation of the two-channels regressor network proposed in Zhong et al. (2018)
    https://hal.archives-ouvertes.fr/hal-01374401/document
    """

    def __init__(self):
        super(TwoChannelsRegressor, self).__init__()

        self.conv1 = nn.Conv2d(2, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 96, kernel_size=3)

        self.fc1 = nn.Linear(1344, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
    

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2, stride=2))
        x = F.relu(F.max_pool2d(self.conv3(x), 3, stride=3))
        x = x.view(-1, 1344)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
    
class SiameseRegressor(nn.Module):
    """
    Implementation of the siamese regressor network proposed in Zhong et al. (2018)
    https://hal.archives-ouvertes.fr/hal-01374401/document
    """

    def __init__(self):
        super(SiameseRegressor, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 96, kernel_size=3)

        self.fc1 = nn.Linear(2688, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
    

    def forward(self, x):
        x1, x2 = x[:,:1], x[:,1:]
        
        x1 = F.relu(F.max_pool2d(self.conv1(x1), 2, stride=2))
        x1 = F.relu(F.max_pool2d(self.conv2(x1), 2, stride=2))
        x1 = F.relu(F.max_pool2d(self.conv3(x1), 3, stride=3))
        x1 = x1.view(-1, 1344)
        
        x2 = F.relu(F.max_pool2d(self.conv1(x2), 2, stride=2))
        x2 = F.relu(F.max_pool2d(self.conv2(x2), 2, stride=2))
        x2 = F.relu(F.max_pool2d(self.conv3(x2), 3, stride=3))
        x2 = x2.view(-1, 1344)
        
        x = torch.cat([x1, x2], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
    
class SiameseClassifier(nn.Module):
    """
    Implementation of the siamese classifier network proposed in Zhong et al. (2018)
    https://hal.archives-ouvertes.fr/hal-01374401/document
    """

    def __init__(self):
        super(SiameseClassifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 96, kernel_size=3)

        self.fc1 = nn.Linear(2688, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)
    

    def forward(self, x):
        x1, x2 = x[:,:1], x[:,1:]
        
        x1 = F.relu(F.max_pool2d(self.conv1(x1), 2, stride=2))
        x1 = F.relu(F.max_pool2d(self.conv2(x1), 2, stride=2))
        x1 = F.relu(F.max_pool2d(self.conv3(x1), 3, stride=3))
        x1 = x1.view(-1, 1344)
        
        x2 = F.relu(F.max_pool2d(self.conv1(x2), 2, stride=2))
        x2 = F.relu(F.max_pool2d(self.conv2(x2), 2, stride=2))
        x2 = F.relu(F.max_pool2d(self.conv3(x2), 3, stride=3))
        x2 = x2.view(-1, 1344)
        
        x = torch.cat([x1, x2], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x