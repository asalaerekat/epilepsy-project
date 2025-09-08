import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

def get_resnet3d(pretrained=True, num_classes=2):
    # 1) load backbone (with weights if true)
    weights = R3D_18_Weights.DEFAULT if pretrained else None
    model = r3d_18(weights=weights)
    
    #for param in model.parameters():
    #    param.requires_grad = False

    # 2) replace head
    in_features    = model.fc.in_features
    model.fc       = nn.Linear(in_features, num_classes)
    
    # 3) freeze everything except layer4 and the new head
    for name, param in model.named_parameters():
        if name.startswith("layer4") or name.startswith("fc"):
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model

def get_vmz(pretrained=True, num_classes=2):
    """
    Returns a placeholder for the VMZ model.
    Implement the loading of the VMZ architecture here.
    """
    raise NotImplementedError("VMZ model integration needs to be implemented.")

def get_model(model_name, pretrained, num_classes):
    if model_name == "resnet3d":
        return get_resnet3d(pretrained, num_classes)
    elif model_name == "vmz":
        return get_vmz(pretrained, num_classes)
    else:
        raise ValueError("Model {} not supported.".format(model_name))
    