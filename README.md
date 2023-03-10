# MPS decomposition of Conv2d models
This repository contains a decomposition method for compressing Conv2d layers of models using Matrix Product States (MPS),
with the goal of reducing inference speed and power consumption of the model. 


## What it does
The method works by decomposing Conv2d layers into two smaller layers using MPS decomposition. 
The amount of compression is controlled by choosing the amount of explained variance to keep from an internally run PCA.

## How to use the method

* load a pretrained model
* decompose with the decompose function
* fine tune for a couple of epochs (1-10 is usually enough from experiments)

### Example usage

```python
from decompose import decompose

model = WideResNet(pretrained=True)

model = decompose(model, pca_ratio=0.8, ignore_list=['initial_conv'])

# Fine tune for a small number of epochs.
```

### Arguments
* `pca_ratio`: The ratio of explained variance to keep from the internal PCA. Takes a number from 0 to 1, with 1 being no compression.
* `ignore_list`: List of strings containing the names of layers to not compress. From experimentation, not compressing the first and very small Conv2d layers were proportionally beneficial.


## Requirements
The required packages can be installed using the following command.
```
pip install -r requirements.txt
```

