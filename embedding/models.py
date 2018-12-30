import torchvision.models as models
import torch.nn as nn

def get_layer(model, i, counter=0):
    """
    Returns the i-th conv2d layer of a model
    If the architecture has nested layers (conv2d inside sequential)
    it returns the first layer by depth-first search
    """
    res = None
    for layer in model.children():
        if res is not None:
            return res, counter
        if isinstance(layer, nn.modules.conv.Conv2d):
            if counter == i:
                return layer, -1
            counter += 1
        else:
            res, counter = get_layer(layer, i, counter=counter)
    return res, counter

resnet50 = models.resnet50(pretrained=True)
resnet50.avgpool = nn.AdaptiveAvgPool2d(1)

resnet50_layer = get_layer(resnet50, 44)[0]
resnet50_channel = 212
