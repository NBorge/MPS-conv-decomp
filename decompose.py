import torch
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import matrix_product_state
from sklearn.decomposition import PCA


def tt_decomposition_conv_layer(layer, pca_ratio):
    weight = layer.weight.data
    weight2D = tl.base.unfold(weight, 0)

    pca = PCA(pca_ratio).fit(weight2D.cpu())
    ranks = int(pca.n_components_)

    last, first = matrix_product_state(weight2D, rank=ranks)
    
    last = last.reshape(weight.shape[0], ranks, 1, 1)
    first = first.reshape(ranks, weight.shape[1], layer.kernel_size[0], layer.kernel_size[1])

    if layer.bias is None:
        bias = False
    else:
        bias = True
    
    layer1 = torch.nn.Conv2d(
            in_channels=first.shape[1], 
            out_channels=first.shape[0], 
            kernel_size=layer.kernel_size, 
            stride=layer.stride, 
            padding=layer.padding, 
            dilation=layer.dilation, 
            bias=bias)

    layer2 = torch.nn.Conv2d(
            in_channels=last.shape[1], 
            out_channels=last.shape[0], 
            kernel_size=1, 
            stride=1,
            padding=0, 
            dilation=layer.dilation, 
            bias=True)

    if bias: 
        layer2.bias.data = layer.bias.data
    layer1.weight.data = first
    layer2.weight.data = last

    new_layers = [layer1, layer2]
    
    return nn.Sequential(*new_layers)


def _decompose(module, pca_ratio, ignore_list):
    if isinstance(module, nn.modules.conv.Conv2d):
        conv_layer = module
        try:
            module = tt_decomposition_conv_layer(conv_layer, pca_ratio)
        except:
            module = module
        return module
    else:
        if len(module._modules) == 0:
            return module
        else:
            for key in module._modules.keys():
                if key not in ignore_list:
                    module._modules[key] = _decompose(module._modules[key], pca_ratio, ignore_list)
            return module


def decompose(model, pca_ratio, ignore_list=[]):
    tl.set_backend('pytorch')
    model = _decompose(model, pca_ratio, ignore_list)
    return model
