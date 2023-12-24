# mlx-cifar-10-cnn
Small CIFAR-10 / CIFAR-100 CNN ResNet-like implementation in Apple mlx, see https://github.com/ml-explore/mlx. 

See https://github.com/menzHSE/torch-cifar-10-cnn for (more or less) the same model being trained using PyTorch.

# Requirements
* Machine with Apple SoC (M1/M2/M3)
* mlx (https://github.com/ml-explore/mlx), tested with version 0.0.6.
* mlx-data (https://github.com/ml-explore/mlx-data)

# Limitations
* mlx does not yet include pooling layers, see https://github.com/ml-explore/mlx/issues/25, so we use strided convolution width `stride=2` for subsampling
* mlx does not yet include batch norm, see https://github.com/ml-explore/mlx/pull/217
* there seem to be some performance issues (with 2D convolutions?) vs. PyTorch with the MPS backend on Apple SoCs, see https://github.com/ml-explore/mlx/issues/243

# Usage
```
$ python train.py -h
usage: Train a simple CNN on CIFAR-10 / CIFAR_100 with mlx. [-h] [--cpu] [--seed SEED] [--batchsize BATCHSIZE] [--epochs EPOCHS]
                                                            [--lr LR] [--dataset {CIFAR-10,CIFAR-100}]

options:
  -h, --help            show this help message and exit
  --cpu                 Use CPU instead of Metal GPU acceleration
  --seed SEED           Random seed
  --batchsize BATCHSIZE
                        Batch size for training
  --epochs EPOCHS       Number of training epochs
  --lr LR               Learning rate
  --dataset {CIFAR-10,CIFAR-100}
                        Select the dataset to use (CIFAR-10 or CIFAR-100)
```

```
python test.py -h
usage: Test a simple CNN on CIFAR-10 / CIFAR-100 with mlx. [-h] [--cpu] --model MODEL [--dataset {CIFAR-10,CIFAR-100}]

options:
  -h, --help            show this help message and exit
  --cpu                 Use CPU instead of Metal GPU acceleration
  --model MODEL         Model filename *.npz
  --dataset {CIFAR-10,CIFAR-100}
                        Select the dataset to use (CIFAR-10 or CIFAR-100)
```

# Examples

## CIFAR-10

### Train on CIFAR-10
`python train.py --dataset=CIFAR-10`

This uses a first gen 16GB Macbook Pro M1. 

```
$ python train.py   --dataset=CIFAR-10 
Options: 
  Device: GPU
  Seed: 0
  Batch size: 64
  Number of epochs: 30
  Learning rate: 0.0003
  Dataset: CIFAR-10
Number of trainable params: 0.5723 M
Starting training ...
Epoch    0: Loss 1.58751 | Train accuracy  0.539 | Test accuracy  0.533 | Throughput     933.11 images/second |  Time   56.130 (s)
Epoch    1: Loss 1.20771 | Train accuracy  0.625 | Test accuracy  0.607 | Throughput     933.00 images/second |  Time   56.044 (s)
Epoch    2: Loss 1.05997 | Train accuracy  0.668 | Test accuracy  0.641 | Throughput     935.98 images/second |  Time   55.864 (s)
Epoch    3: Loss 0.96530 | Train accuracy  0.701 | Test accuracy  0.670 | Throughput     933.35 images/second |  Time   56.084 (s)
Epoch    4: Loss 0.89180 | Train accuracy  0.727 | Test accuracy  0.691 | Throughput     932.51 images/second |  Time   56.115 (s)
Epoch    5: Loss 0.83689 | Train accuracy  0.748 | Test accuracy  0.705 | Throughput     930.56 images/second |  Time   56.279 (s)
Epoch    6: Loss 0.78752 | Train accuracy  0.761 | Test accuracy  0.716 | Throughput     931.91 images/second |  Time   56.168 (s)
Epoch    7: Loss 0.74867 | Train accuracy  0.775 | Test accuracy  0.727 | Throughput     931.87 images/second |  Time   56.164 (s)
Epoch    8: Loss 0.71237 | Train accuracy  0.785 | Test accuracy  0.731 | Throughput     931.34 images/second |  Time   56.165 (s)
Epoch    9: Loss 0.67609 | Train accuracy  0.803 | Test accuracy  0.742 | Throughput     930.87 images/second |  Time   56.221 (s)
Epoch   10: Loss 0.64472 | Train accuracy  0.808 | Test accuracy  0.749 | Throughput     931.13 images/second |  Time   56.187 (s)
Epoch   11: Loss 0.61530 | Train accuracy  0.822 | Test accuracy  0.756 | Throughput     932.78 images/second |  Time   56.100 (s)
Epoch   12: Loss 0.59308 | Train accuracy  0.830 | Test accuracy  0.757 | Throughput     929.90 images/second |  Time   56.277 (s)
Epoch   13: Loss 0.57011 | Train accuracy  0.838 | Test accuracy  0.762 | Throughput     949.81 images/second |  Time   55.166 (s)
Epoch   14: Loss 0.54330 | Train accuracy  0.849 | Test accuracy  0.771 | Throughput     929.69 images/second |  Time   56.269 (s)
Epoch   15: Loss 0.52460 | Train accuracy  0.860 | Test accuracy  0.777 | Throughput     928.88 images/second |  Time   56.307 (s)
Epoch   16: Loss 0.50588 | Train accuracy  0.865 | Test accuracy  0.776 | Throughput     934.33 images/second |  Time   55.993 (s)
Epoch   17: Loss 0.49043 | Train accuracy  0.872 | Test accuracy  0.776 | Throughput     933.06 images/second |  Time   56.090 (s)
Epoch   18: Loss 0.47078 | Train accuracy  0.880 | Test accuracy  0.781 | Throughput     927.28 images/second |  Time   56.424 (s)
Epoch   19: Loss 0.46172 | Train accuracy  0.889 | Test accuracy  0.785 | Throughput     932.43 images/second |  Time   56.086 (s)
Epoch   20: Loss 0.44215 | Train accuracy  0.895 | Test accuracy  0.787 | Throughput     932.54 images/second |  Time   56.130 (s)
Epoch   21: Loss 0.42690 | Train accuracy  0.893 | Test accuracy  0.784 | Throughput     930.14 images/second |  Time   56.232 (s)
Epoch   22: Loss 0.41342 | Train accuracy  0.897 | Test accuracy  0.782 | Throughput     931.18 images/second |  Time   56.184 (s)
Epoch   23: Loss 0.40281 | Train accuracy  0.908 | Test accuracy  0.788 | Throughput     929.88 images/second |  Time   56.251 (s)
Epoch   24: Loss 0.38836 | Train accuracy  0.912 | Test accuracy  0.787 | Throughput     930.10 images/second |  Time   56.276 (s)
Epoch   25: Loss 0.38196 | Train accuracy  0.910 | Test accuracy  0.789 | Throughput     931.56 images/second |  Time   56.145 (s)
Epoch   26: Loss 0.37404 | Train accuracy  0.921 | Test accuracy  0.792 | Throughput     930.78 images/second |  Time   56.198 (s)
Epoch   27: Loss 0.36139 | Train accuracy  0.923 | Test accuracy  0.788 | Throughput     930.01 images/second |  Time   56.264 (s)
Epoch   28: Loss 0.35089 | Train accuracy  0.921 | Test accuracy  0.784 | Throughput     931.65 images/second |  Time   56.154 (s)
Epoch   29: Loss 0.34075 | Train accuracy  0.933 | Test accuracy  0.791 | Throughput     931.90 images/second |  Time   56.139 (s)
```

### Test on CIFAR-10
```
$ python test.py  --model models/model_CIFAR-10_029.npz
Loaded model for CIFAR-10 from  models/model_CIFAR-10_029.npz
Starting testing ...
....
Test accuracy: 0.7912495133399963
```

## CIFAR-100

### Train on CIFAR-100
`python train.py --dataset CIFAR-100`

This uses a first gen 16GB Macbook Pro M1. 

```
$ python train.py  --dataset=CIFAR-100
Options: 
  Device: GPU
  Seed: 0
  Batch size: 64
  Number of epochs: 30
  Learning rate: 0.0003
  Dataset: CIFAR-100
Number of trainable params: 0.5839 M
Starting training ...
Epoch    0: Loss 3.94354 | Train accuracy  0.168 | Test accuracy  0.159 | Throughput     939.33 images/second |  Time   55.598 (s)
Epoch    1: Loss 3.29697 | Train accuracy  0.251 | Test accuracy  0.240 | Throughput     934.35 images/second |  Time   55.906 (s)
Epoch    2: Loss 2.98886 | Train accuracy  0.300 | Test accuracy  0.285 | Throughput     940.89 images/second |  Time   55.574 (s)
Epoch    3: Loss 2.79632 | Train accuracy  0.340 | Test accuracy  0.314 | Throughput     939.58 images/second |  Time   55.710 (s)
Epoch    4: Loss 2.65143 | Train accuracy  0.371 | Test accuracy  0.330 | Throughput     934.43 images/second |  Time   55.972 (s)
Epoch    5: Loss 2.52931 | Train accuracy  0.394 | Test accuracy  0.349 | Throughput     932.86 images/second |  Time   56.099 (s)
Epoch    6: Loss 2.43186 | Train accuracy  0.416 | Test accuracy  0.365 | Throughput     929.37 images/second |  Time   56.275 (s)
Epoch    7: Loss 2.34618 | Train accuracy  0.436 | Test accuracy  0.374 | Throughput     932.51 images/second |  Time   56.114 (s)
Epoch    8: Loss 2.27019 | Train accuracy  0.454 | Test accuracy  0.384 | Throughput     931.55 images/second |  Time   56.167 (s)
Epoch    9: Loss 2.20327 | Train accuracy  0.470 | Test accuracy  0.390 | Throughput     931.12 images/second |  Time   56.179 (s)
Epoch   10: Loss 2.13615 | Train accuracy  0.487 | Test accuracy  0.402 | Throughput     929.37 images/second |  Time   56.291 (s)
Epoch   11: Loss 2.07444 | Train accuracy  0.502 | Test accuracy  0.408 | Throughput     928.65 images/second |  Time   56.354 (s)
Epoch   12: Loss 2.02129 | Train accuracy  0.520 | Test accuracy  0.415 | Throughput     928.31 images/second |  Time   56.324 (s)
Epoch   13: Loss 1.96446 | Train accuracy  0.529 | Test accuracy  0.415 | Throughput     930.53 images/second |  Time   56.242 (s)
Epoch   14: Loss 1.91196 | Train accuracy  0.544 | Test accuracy  0.425 | Throughput     929.58 images/second |  Time   56.277 (s)
Epoch   15: Loss 1.86621 | Train accuracy  0.555 | Test accuracy  0.427 | Throughput     928.72 images/second |  Time   56.314 (s)
Epoch   16: Loss 1.82444 | Train accuracy  0.574 | Test accuracy  0.438 | Throughput     943.28 images/second |  Time   55.533 (s)
Epoch   17: Loss 1.77800 | Train accuracy  0.586 | Test accuracy  0.446 | Throughput     929.21 images/second |  Time   56.293 (s)
Epoch   18: Loss 1.73693 | Train accuracy  0.600 | Test accuracy  0.446 | Throughput     930.99 images/second |  Time   56.206 (s)
Epoch   19: Loss 1.70420 | Train accuracy  0.607 | Test accuracy  0.450 | Throughput     939.47 images/second |  Time   55.673 (s)
Epoch   20: Loss 1.66383 | Train accuracy  0.621 | Test accuracy  0.450 | Throughput     932.04 images/second |  Time   56.137 (s)
Epoch   21: Loss 1.62749 | Train accuracy  0.630 | Test accuracy  0.454 | Throughput     930.05 images/second |  Time   56.253 (s)
Epoch   22: Loss 1.59953 | Train accuracy  0.642 | Test accuracy  0.458 | Throughput     929.75 images/second |  Time   56.281 (s)
Epoch   23: Loss 1.56418 | Train accuracy  0.645 | Test accuracy  0.456 | Throughput     930.32 images/second |  Time   56.283 (s)
Epoch   24: Loss 1.52951 | Train accuracy  0.655 | Test accuracy  0.457 | Throughput     928.47 images/second |  Time   56.385 (s)
Epoch   25: Loss 1.50257 | Train accuracy  0.671 | Test accuracy  0.463 | Throughput     926.21 images/second |  Time   56.488 (s)
Epoch   26: Loss 1.48180 | Train accuracy  0.675 | Test accuracy  0.464 | Throughput     928.57 images/second |  Time   56.338 (s)
Epoch   27: Loss 1.45204 | Train accuracy  0.678 | Test accuracy  0.463 | Throughput     932.04 images/second |  Time   56.133 (s)
Epoch   28: Loss 1.42580 | Train accuracy  0.691 | Test accuracy  0.471 | Throughput     927.45 images/second |  Time   56.682 (s)
Epoch   29: Loss 1.40150 | Train accuracy  0.697 | Test accuracy  0.468 | Throughput     933.55 images/second |  Time   56.095 (s)

```

### Test on CIFAR-100
```
$ python test.py  --model models/model_CIFAR-100_029.npz --dataset=CIFAR-100
Loaded model for CIFAR-100 from models/model_CIFAR-100_029.npz
Starting testing ...
....
Test accuracy: 0.46810701847076416
```
