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
Number of trainable params: 0.5732 M
Starting training ...
Epoch    0: Loss 1.37323 | Train accuracy  0.627 | Test accuracy  0.615 | Throughput     710.90 images/second |  Time   72.685 (s)
Epoch    1: Loss 1.00935 | Train accuracy  0.703 | Test accuracy  0.682 | Throughput     762.42 images/second |  Time   67.574 (s)
Epoch    2: Loss 0.86681 | Train accuracy  0.747 | Test accuracy  0.714 | Throughput     699.50 images/second |  Time   73.813 (s)
Epoch    3: Loss 0.77643 | Train accuracy  0.774 | Test accuracy  0.733 | Throughput     745.52 images/second |  Time   69.039 (s)
Epoch    4: Loss 0.71323 | Train accuracy  0.790 | Test accuracy  0.751 | Throughput     783.78 images/second |  Time   65.545 (s)
Epoch    5: Loss 0.65958 | Train accuracy  0.805 | Test accuracy  0.755 | Throughput     783.42 images/second |  Time   65.090 (s)
Epoch    6: Loss 0.61841 | Train accuracy  0.812 | Test accuracy  0.755 | Throughput     749.61 images/second |  Time   68.736 (s)
Epoch    7: Loss 0.58169 | Train accuracy  0.833 | Test accuracy  0.769 | Throughput     773.81 images/second |  Time   66.379 (s)
Epoch    8: Loss 0.54770 | Train accuracy  0.847 | Test accuracy  0.780 | Throughput     723.40 images/second |  Time   71.381 (s)
Epoch    9: Loss 0.52019 | Train accuracy  0.859 | Test accuracy  0.783 | Throughput     777.75 images/second |  Time   66.101 (s)
Epoch   10: Loss 0.50216 | Train accuracy  0.868 | Test accuracy  0.792 | Throughput     756.67 images/second |  Time   67.946 (s)
Epoch   11: Loss 0.47864 | Train accuracy  0.879 | Test accuracy  0.796 | Throughput     777.26 images/second |  Time   66.094 (s)
Epoch   12: Loss 0.45690 | Train accuracy  0.882 | Test accuracy  0.797 | Throughput     782.12 images/second |  Time   65.511 (s)
Epoch   13: Loss 0.44128 | Train accuracy  0.889 | Test accuracy  0.802 | Throughput     815.26 images/second |  Time   62.345 (s)
Epoch   14: Loss 0.42271 | Train accuracy  0.897 | Test accuracy  0.800 | Throughput     762.10 images/second |  Time   67.700 (s)
Epoch   15: Loss 0.40843 | Train accuracy  0.904 | Test accuracy  0.803 | Throughput     699.48 images/second |  Time   73.766 (s)
Epoch   16: Loss 0.39179 | Train accuracy  0.911 | Test accuracy  0.806 | Throughput     731.39 images/second |  Time   70.605 (s)
Epoch   17: Loss 0.37631 | Train accuracy  0.915 | Test accuracy  0.811 | Throughput     790.64 images/second |  Time   64.510 (s)
Epoch   18: Loss 0.37233 | Train accuracy  0.910 | Test accuracy  0.803 | Throughput     703.67 images/second |  Time   73.299 (s)
Epoch   19: Loss 0.35850 | Train accuracy  0.920 | Test accuracy  0.808 | Throughput     702.39 images/second |  Time   73.443 (s)
Epoch   20: Loss 0.34254 | Train accuracy  0.926 | Test accuracy  0.808 | Throughput     791.19 images/second |  Time   64.739 (s)
Epoch   21: Loss 0.33144 | Train accuracy  0.932 | Test accuracy  0.810 | Throughput     805.78 images/second |  Time   63.179 (s)
Epoch   22: Loss 0.31811 | Train accuracy  0.933 | Test accuracy  0.810 | Throughput     810.89 images/second |  Time   62.729 (s)
Epoch   23: Loss 0.31665 | Train accuracy  0.937 | Test accuracy  0.813 | Throughput     805.33 images/second |  Time   63.312 (s)
Epoch   24: Loss 0.30711 | Train accuracy  0.941 | Test accuracy  0.814 | Throughput     759.31 images/second |  Time   67.818 (s)
Epoch   25: Loss 0.29356 | Train accuracy  0.945 | Test accuracy  0.814 | Throughput     706.23 images/second |  Time   73.112 (s)
Epoch   26: Loss 0.28890 | Train accuracy  0.946 | Test accuracy  0.813 | Throughput     718.40 images/second |  Time   71.903 (s)
Epoch   27: Loss 0.28621 | Train accuracy  0.948 | Test accuracy  0.815 | Throughput     754.54 images/second |  Time   68.263 (s)
Epoch   28: Loss 0.27018 | Train accuracy  0.949 | Test accuracy  0.816 | Throughput     807.83 images/second |  Time   63.029 (s)
Epoch   29: Loss 0.27048 | Train accuracy  0.948 | Test accuracy  0.816 | Throughput     815.62 images/second |  Time   62.358 (s)
```

### Test on CIFAR-10
```
$ python test.py  --model models/model_CIFAR-10_029.npz
Loaded model for CIFAR-10 from  models/model_CIFAR-10_029.npz
Starting testing ...
....
Test accuracy: 0.8162495133399963
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
Number of trainable params: 0.5848 M
Starting training ...
Epoch    0: Loss 3.61510 | Train accuracy  0.251 | Test accuracy  0.238 | Throughput     809.27 images/second |  Time   62.879 (s)
Epoch    1: Loss 2.93408 | Train accuracy  0.336 | Test accuracy  0.310 | Throughput     783.02 images/second |  Time   65.187 (s)
Epoch    2: Loss 2.60984 | Train accuracy  0.390 | Test accuracy  0.353 | Throughput     779.75 images/second |  Time   65.471 (s)
Epoch    3: Loss 2.39885 | Train accuracy  0.432 | Test accuracy  0.381 | Throughput     782.64 images/second |  Time   65.217 (s)
Epoch    4: Loss 2.23692 | Train accuracy  0.470 | Test accuracy  0.406 | Throughput     791.60 images/second |  Time   64.457 (s)
Epoch    5: Loss 2.11503 | Train accuracy  0.494 | Test accuracy  0.423 | Throughput     781.21 images/second |  Time   65.348 (s)
Epoch    6: Loss 2.01315 | Train accuracy  0.520 | Test accuracy  0.438 | Throughput     780.80 images/second |  Time   65.379 (s)
Epoch    7: Loss 1.92189 | Train accuracy  0.541 | Test accuracy  0.446 | Throughput     788.81 images/second |  Time   64.719 (s)
Epoch    8: Loss 1.84478 | Train accuracy  0.560 | Test accuracy  0.458 | Throughput     780.71 images/second |  Time   65.374 (s)
Epoch    9: Loss 1.77813 | Train accuracy  0.576 | Test accuracy  0.465 | Throughput     778.45 images/second |  Time   65.599 (s)
Epoch   10: Loss 1.72146 | Train accuracy  0.597 | Test accuracy  0.475 | Throughput     778.89 images/second |  Time   65.559 (s)
Epoch   11: Loss 1.66444 | Train accuracy  0.611 | Test accuracy  0.477 | Throughput     773.94 images/second |  Time   65.983 (s)
Epoch   12: Loss 1.61021 | Train accuracy  0.625 | Test accuracy  0.484 | Throughput     780.69 images/second |  Time   65.397 (s)
Epoch   13: Loss 1.56243 | Train accuracy  0.637 | Test accuracy  0.486 | Throughput     779.19 images/second |  Time   65.543 (s)
Epoch   14: Loss 1.52759 | Train accuracy  0.647 | Test accuracy  0.489 | Throughput     779.82 images/second |  Time   65.475 (s)
Epoch   15: Loss 1.48773 | Train accuracy  0.662 | Test accuracy  0.499 | Throughput     779.08 images/second |  Time   65.548 (s)
Epoch   16: Loss 1.44502 | Train accuracy  0.671 | Test accuracy  0.497 | Throughput     779.61 images/second |  Time   65.499 (s)
Epoch   17: Loss 1.41484 | Train accuracy  0.684 | Test accuracy  0.500 | Throughput     779.42 images/second |  Time   65.522 (s)
Epoch   18: Loss 1.38248 | Train accuracy  0.694 | Test accuracy  0.506 | Throughput     781.51 images/second |  Time   65.337 (s)
Epoch   19: Loss 1.35413 | Train accuracy  0.700 | Test accuracy  0.506 | Throughput     779.50 images/second |  Time   65.518 (s)
Epoch   20: Loss 1.32103 | Train accuracy  0.711 | Test accuracy  0.510 | Throughput     783.19 images/second |  Time   65.208 (s)
Epoch   21: Loss 1.28973 | Train accuracy  0.722 | Test accuracy  0.504 | Throughput     779.00 images/second |  Time   65.544 (s)
Epoch   22: Loss 1.27353 | Train accuracy  0.728 | Test accuracy  0.515 | Throughput     781.26 images/second |  Time   65.364 (s)
Epoch   23: Loss 1.24663 | Train accuracy  0.733 | Test accuracy  0.515 | Throughput     778.15 images/second |  Time   65.627 (s)
Epoch   24: Loss 1.21778 | Train accuracy  0.740 | Test accuracy  0.514 | Throughput     779.01 images/second |  Time   65.553 (s)
Epoch   25: Loss 1.19863 | Train accuracy  0.755 | Test accuracy  0.514 | Throughput     778.93 images/second |  Time   65.553 (s)
Epoch   26: Loss 1.17385 | Train accuracy  0.762 | Test accuracy  0.518 | Throughput     778.15 images/second |  Time   65.637 (s)
Epoch   27: Loss 1.16084 | Train accuracy  0.762 | Test accuracy  0.514 | Throughput     778.25 images/second |  Time   65.626 (s)
Epoch   28: Loss 1.13944 | Train accuracy  0.772 | Test accuracy  0.519 | Throughput     779.78 images/second |  Time   65.483 (s)
Epoch   29: Loss 1.11530 | Train accuracy  0.776 | Test accuracy  0.517 | Throughput     774.04 images/second |  Time   66.198 (s)

```

### Test on CIFAR-100
```
$ python test.py  --model models/model_CIFAR-100_029.npz --dataset=CIFAR-100
Loaded model for CIFAR-100 from models/model_CIFAR-100_029.npz
Starting testing ...
....
Test accuracy: 0.51710701847076416
```
