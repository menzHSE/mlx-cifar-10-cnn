# mlx-cifar-10-cnn
Small CIFAR-10 / CIFAR-100 CNN implementation in Apple mlx. Used to try-out the first public version of https://github.com/ml-explore/mlx. It heavily borrows from the CIFAR example in https://github.com/ml-explore/mlx-examples

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
Number of trainable params: 0.5506 M
Starting training ...
Epoch    0: Loss 1.72772 | Train accuracy  0.473 | Test accuracy  0.473 | Throughput    1270.61 images/second |  Time   40.551 (s)
Epoch    1: Loss 1.38787 | Train accuracy  0.551 | Test accuracy  0.546 | Throughput    1225.06 images/second |  Time   42.347 (s)
Epoch    2: Loss 1.24764 | Train accuracy  0.596 | Test accuracy  0.588 | Throughput    1218.43 images/second |  Time   42.610 (s)
Epoch    3: Loss 1.15085 | Train accuracy  0.629 | Test accuracy  0.615 | Throughput    1216.17 images/second |  Time   42.685 (s)
Epoch    4: Loss 1.08930 | Train accuracy  0.653 | Test accuracy  0.641 | Throughput    1216.78 images/second |  Time   42.674 (s)
Epoch    5: Loss 1.03501 | Train accuracy  0.670 | Test accuracy  0.651 | Throughput    1223.72 images/second |  Time   42.787 (s)
Epoch    6: Loss 0.99111 | Train accuracy  0.682 | Test accuracy  0.658 | Throughput    1218.57 images/second |  Time   42.598 (s)
Epoch    7: Loss 0.94907 | Train accuracy  0.702 | Test accuracy  0.674 | Throughput    1216.33 images/second |  Time   42.690 (s)
Epoch    8: Loss 0.92009 | Train accuracy  0.716 | Test accuracy  0.686 | Throughput    1216.33 images/second |  Time   42.691 (s)
Epoch    9: Loss 0.88479 | Train accuracy  0.724 | Test accuracy  0.695 | Throughput    1189.44 images/second |  Time   44.431 (s)
Epoch   10: Loss 0.85970 | Train accuracy  0.739 | Test accuracy  0.707 | Throughput    1156.15 images/second |  Time   45.419 (s)
Epoch   11: Loss 0.83244 | Train accuracy  0.741 | Test accuracy  0.709 | Throughput    1200.47 images/second |  Time   43.423 (s)
Epoch   12: Loss 0.80880 | Train accuracy  0.750 | Test accuracy  0.719 | Throughput    1212.16 images/second |  Time   42.863 (s)
Epoch   13: Loss 0.79200 | Train accuracy  0.761 | Test accuracy  0.726 | Throughput    1221.41 images/second |  Time   42.526 (s)
Epoch   14: Loss 0.76958 | Train accuracy  0.766 | Test accuracy  0.727 | Throughput    1211.73 images/second |  Time   42.883 (s)
Epoch   15: Loss 0.74821 | Train accuracy  0.781 | Test accuracy  0.742 | Throughput    1211.81 images/second |  Time   42.877 (s)
Epoch   16: Loss 0.73170 | Train accuracy  0.778 | Test accuracy  0.738 | Throughput    1210.92 images/second |  Time   42.913 (s)
Epoch   17: Loss 0.71408 | Train accuracy  0.788 | Test accuracy  0.741 | Throughput    1212.25 images/second |  Time   42.860 (s)
Epoch   18: Loss 0.69373 | Train accuracy  0.796 | Test accuracy  0.751 | Throughput    1209.89 images/second |  Time   42.947 (s)
Epoch   19: Loss 0.68047 | Train accuracy  0.805 | Test accuracy  0.762 | Throughput    1207.62 images/second |  Time   43.037 (s)
Epoch   20: Loss 0.66610 | Train accuracy  0.814 | Test accuracy  0.764 | Throughput    1209.67 images/second |  Time   42.948 (s)
Epoch   21: Loss 0.64284 | Train accuracy  0.816 | Test accuracy  0.762 | Throughput    1208.19 images/second |  Time   43.006 (s)
Epoch   22: Loss 0.63892 | Train accuracy  0.817 | Test accuracy  0.765 | Throughput    1212.51 images/second |  Time   42.847 (s)
Epoch   23: Loss 0.62249 | Train accuracy  0.827 | Test accuracy  0.772 | Throughput    1208.26 images/second |  Time   43.011 (s)
Epoch   24: Loss 0.60894 | Train accuracy  0.834 | Test accuracy  0.779 | Throughput    1210.83 images/second |  Time   42.900 (s)
Epoch   25: Loss 0.59800 | Train accuracy  0.837 | Test accuracy  0.778 | Throughput    1208.11 images/second |  Time   43.017 (s)
Epoch   26: Loss 0.58398 | Train accuracy  0.844 | Test accuracy  0.781 | Throughput    1208.90 images/second |  Time   42.980 (s)
Epoch   27: Loss 0.57689 | Train accuracy  0.843 | Test accuracy  0.778 | Throughput    1209.56 images/second |  Time   42.958 (s)
Epoch   28: Loss 0.56539 | Train accuracy  0.851 | Test accuracy  0.783 | Throughput    1218.75 images/second |  Time   42.567 (s)
Epoch   29: Loss 0.54964 | Train accuracy  0.851 | Test accuracy  0.782 | Throughput    1261.20 images/second |  Time   40.929 (s)
```

### Test on CIFAR-10
```
$ python test.py  --model models/model_CIFAR-10_029.npz
Loaded model for CIFAR-10 from  models/model_CIFAR-10_029.npz
Starting testing ...
....
Test accuracy: 0.7815495133399963
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
Number of trainable params: 0.5622 M
Starting training ...
Epoch    0: Loss 4.13097 | Train accuracy  0.125 | Test accuracy  0.119 | Throughput    1219.93 images/second |  Time   42.611 (s)
Epoch    1: Loss 3.58333 | Train accuracy  0.183 | Test accuracy  0.179 | Throughput    1227.56 images/second |  Time   42.218 (s)
Epoch    2: Loss 3.32894 | Train accuracy  0.226 | Test accuracy  0.224 | Throughput    1231.47 images/second |  Time   42.069 (s)
Epoch    3: Loss 3.16587 | Train accuracy  0.260 | Test accuracy  0.247 | Throughput    1228.64 images/second |  Time   42.154 (s)
Epoch    4: Loss 3.04211 | Train accuracy  0.286 | Test accuracy  0.272 | Throughput    1227.03 images/second |  Time   42.209 (s)
Epoch    5: Loss 2.94239 | Train accuracy  0.301 | Test accuracy  0.285 | Throughput    1226.65 images/second |  Time   42.239 (s)
Epoch    6: Loss 2.85935 | Train accuracy  0.322 | Test accuracy  0.299 | Throughput    1227.32 images/second |  Time   42.205 (s)
Epoch    7: Loss 2.78884 | Train accuracy  0.341 | Test accuracy  0.313 | Throughput    1226.52 images/second |  Time   42.237 (s)
Epoch    8: Loss 2.72798 | Train accuracy  0.349 | Test accuracy  0.324 | Throughput    1226.55 images/second |  Time   42.223 (s)
Epoch    9: Loss 2.65516 | Train accuracy  0.366 | Test accuracy  0.336 | Throughput    1238.99 images/second |  Time   41.737 (s)
Epoch   10: Loss 2.60537 | Train accuracy  0.382 | Test accuracy  0.345 | Throughput    1247.58 images/second |  Time   41.414 (s)
Epoch   11: Loss 2.55005 | Train accuracy  0.390 | Test accuracy  0.352 | Throughput    1250.44 images/second |  Time   41.288 (s)
Epoch   12: Loss 2.50623 | Train accuracy  0.403 | Test accuracy  0.359 | Throughput    1235.72 images/second |  Time   41.887 (s)
Epoch   13: Loss 2.45849 | Train accuracy  0.421 | Test accuracy  0.368 | Throughput    1233.92 images/second |  Time   41.965 (s)
Epoch   14: Loss 2.42022 | Train accuracy  0.431 | Test accuracy  0.379 | Throughput    1230.91 images/second |  Time   42.068 (s)
Epoch   15: Loss 2.38206 | Train accuracy  0.440 | Test accuracy  0.385 | Throughput    1254.49 images/second |  Time   41.143 (s)
Epoch   16: Loss 2.33949 | Train accuracy  0.450 | Test accuracy  0.390 | Throughput    1225.65 images/second |  Time   42.284 (s)
Epoch   17: Loss 2.30976 | Train accuracy  0.456 | Test accuracy  0.391 | Throughput    1239.16 images/second |  Time   41.761 (s)
Epoch   18: Loss 2.26624 | Train accuracy  0.467 | Test accuracy  0.398 | Throughput    1227.50 images/second |  Time   42.213 (s)
Epoch   19: Loss 2.23414 | Train accuracy  0.469 | Test accuracy  0.398 | Throughput    1227.06 images/second |  Time   42.237 (s)
Epoch   20: Loss 2.20615 | Train accuracy  0.483 | Test accuracy  0.408 | Throughput    1224.38 images/second |  Time   42.347 (s)
Epoch   21: Loss 2.17266 | Train accuracy  0.490 | Test accuracy  0.408 | Throughput    1227.01 images/second |  Time   42.241 (s)
Epoch   22: Loss 2.14304 | Train accuracy  0.502 | Test accuracy  0.415 | Throughput    1224.37 images/second |  Time   42.346 (s)
Epoch   23: Loss 2.11957 | Train accuracy  0.506 | Test accuracy  0.419 | Throughput    1224.29 images/second |  Time   42.353 (s)
Epoch   24: Loss 2.08617 | Train accuracy  0.516 | Test accuracy  0.426 | Throughput    1225.07 images/second |  Time   42.326 (s)
Epoch   25: Loss 2.06267 | Train accuracy  0.515 | Test accuracy  0.424 | Throughput    1231.51 images/second |  Time   42.074 (s)
Epoch   26: Loss 2.03589 | Train accuracy  0.520 | Test accuracy  0.425 | Throughput    1222.18 images/second |  Time   42.465 (s)
Epoch   27: Loss 2.01222 | Train accuracy  0.533 | Test accuracy  0.429 | Throughput    1233.59 images/second |  Time   41.998 (s)
Epoch   28: Loss 1.99446 | Train accuracy  0.536 | Test accuracy  0.433 | Throughput    1247.90 images/second |  Time   41.708 (s)
Epoch   29: Loss 1.96215 | Train accuracy  0.550 | Test accuracy  0.433 | Throughput    1256.37 images/second |  Time   41.311 (s)

```

### Test on CIFAR-100
```
$ python test.py  --model models/model_CIFAR-100_029.npz --dataset=CIFAR-100
Loaded model for CIFAR-100 from models/model_CIFAR-100_029.npz
Starting testing ...
....
Test accuracy: 0.43310701847076416
```
