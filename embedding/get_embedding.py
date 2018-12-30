import torchvision.models as models
import torch
import torch.nn as nn


def get_embedding(model, layer, channel, tensor):
    """
    return an embedding for a given image stored in a pytorch tensor
    computed with the result of a given channel of a given layer of 
    the model
    """
    model.eval()

    tensor = torch.cat((tensor.unsqueeze(0), tensor.unsqueeze(0), tensor.unsqueeze(0)), 1)

    embedding = []
    def get_output(m, i, o):
        embedding.append(o.detach())
    
    h = layer.register_forward_hook(get_output)

    model(tensor)

    h.remove()

    return embedding[0][:, channel]


