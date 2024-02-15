# mlx-cifar-10-cnn
Small CIFAR-10 / CIFAR-100 CNN ResNet-like implementation in Apple MLX, see https://github.com/ml-explore/mlx. This is based on the CIFAR example in [MLX-examples](https://github.com/ml-explore/mlx-examples/tree/main/cifar). 

See https://github.com/menzHSE/torch-cifar-10-cnn for (more or less) the same model being trained using PyTorch.

# Requirements
* Machine with Apple SoC (M1/M2/M3)
* mlx (https://github.com/ml-explore/mlx), tested with version 0.0.6.
* mlx-data (https://github.com/ml-explore/mlx-data)

# Limitations
* mlx does not yet include pooling layers, see https://github.com/ml-explore/mlx/issues/25, so we use strided convolution width `stride=2` for subsampling
* there seem to be some performance issues (with 2D convolutions?) vs. PyTorch with the MPS backend on Apple SoCs, see https://github.com/ml-explore/mlx/issues/243

# Usage
```
$ python train.py -h
usage: train.py [-h] [--cpu] [--seed SEED] [--batchsize BATCHSIZE] [--epochs EPOCHS] [--lr LR]
                [--dataset {CIFAR-10,CIFAR-100}]

Train a simple CNN on CIFAR-10 / CIFAR_100 with mlx.

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
$ python test.py -h
usage: test.py [-h] [--cpu] --model MODEL [--dataset {CIFAR-10,CIFAR-100}]

Test a simple CNN on CIFAR-10 / CIFAR-100 with mlx.

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
Epoch    0: Loss 1.26400 | Train accuracy  0.677 | Test accuracy  0.668 | Throughput     604.77 images/second |  Time   84.664 (s)
Epoch    1: Loss 0.84543 | Train accuracy  0.752 | Test accuracy  0.734 | Throughput     617.74 images/second |  Time   82.696 (s)
Epoch    2: Loss 0.70719 | Train accuracy  0.784 | Test accuracy  0.761 | Throughput     634.06 images/second |  Time   80.222 (s)
Epoch    3: Loss 0.62693 | Train accuracy  0.803 | Test accuracy  0.775 | Throughput     634.46 images/second |  Time   80.151 (s)
Epoch    4: Loss 0.56904 | Train accuracy  0.832 | Test accuracy  0.799 | Throughput     633.36 images/second |  Time   80.313 (s)
Epoch    5: Loss 0.52295 | Train accuracy  0.847 | Test accuracy  0.805 | Throughput     637.70 images/second |  Time   79.600 (s)
Epoch    6: Loss 0.48411 | Train accuracy  0.862 | Test accuracy  0.818 | Throughput     643.29 images/second |  Time   78.954 (s)
Epoch    7: Loss 0.45588 | Train accuracy  0.869 | Test accuracy  0.820 | Throughput     646.63 images/second |  Time   78.496 (s)
Epoch    8: Loss 0.42985 | Train accuracy  0.879 | Test accuracy  0.828 | Throughput     644.11 images/second |  Time   78.820 (s)
Epoch    9: Loss 0.40379 | Train accuracy  0.890 | Test accuracy  0.830 | Throughput     637.39 images/second |  Time   79.745 (s)
Epoch   10: Loss 0.38491 | Train accuracy  0.898 | Test accuracy  0.834 | Throughput     634.22 images/second |  Time   80.209 (s)
Epoch   11: Loss 0.36022 | Train accuracy  0.905 | Test accuracy  0.840 | Throughput     633.75 images/second |  Time   80.247 (s)
Epoch   12: Loss 0.34720 | Train accuracy  0.910 | Test accuracy  0.835 | Throughput     634.65 images/second |  Time   80.141 (s)
Epoch   13: Loss 0.33135 | Train accuracy  0.920 | Test accuracy  0.850 | Throughput     633.73 images/second |  Time   80.269 (s)
Epoch   14: Loss 0.31703 | Train accuracy  0.917 | Test accuracy  0.845 | Throughput     634.39 images/second |  Time   80.179 (s)
Epoch   15: Loss 0.30135 | Train accuracy  0.927 | Test accuracy  0.849 | Throughput     634.51 images/second |  Time   80.166 (s)
Epoch   16: Loss 0.29303 | Train accuracy  0.936 | Test accuracy  0.848 | Throughput     634.22 images/second |  Time   80.196 (s)
Epoch   17: Loss 0.27962 | Train accuracy  0.937 | Test accuracy  0.851 | Throughput     634.59 images/second |  Time   80.142 (s)
Epoch   18: Loss 0.26781 | Train accuracy  0.938 | Test accuracy  0.851 | Throughput     635.68 images/second |  Time   80.002 (s)
Epoch   19: Loss 0.25911 | Train accuracy  0.942 | Test accuracy  0.851 | Throughput     634.28 images/second |  Time   80.189 (s)
Epoch   20: Loss 0.24879 | Train accuracy  0.945 | Test accuracy  0.856 | Throughput     634.25 images/second |  Time   80.197 (s)
Epoch   21: Loss 0.23914 | Train accuracy  0.949 | Test accuracy  0.854 | Throughput     634.19 images/second |  Time   80.194 (s)
Epoch   22: Loss 0.23205 | Train accuracy  0.949 | Test accuracy  0.855 | Throughput     633.79 images/second |  Time   80.266 (s)
Epoch   23: Loss 0.22816 | Train accuracy  0.955 | Test accuracy  0.856 | Throughput     633.53 images/second |  Time   80.283 (s)
Epoch   24: Loss 0.21693 | Train accuracy  0.961 | Test accuracy  0.861 | Throughput     634.43 images/second |  Time   80.168 (s)
Epoch   25: Loss 0.21044 | Train accuracy  0.952 | Test accuracy  0.854 | Throughput     631.70 images/second |  Time   80.545 (s)
Epoch   26: Loss 0.20578 | Train accuracy  0.963 | Test accuracy  0.863 | Throughput     634.65 images/second |  Time   80.139 (s)
Epoch   27: Loss 0.20007 | Train accuracy  0.966 | Test accuracy  0.863 | Throughput     634.75 images/second |  Time   80.122 (s)
Epoch   28: Loss 0.19409 | Train accuracy  0.966 | Test accuracy  0.864 | Throughput     632.30 images/second |  Time   80.465 (s)
Epoch   29: Loss 0.18525 | Train accuracy  0.968 | Test accuracy  0.861 | Throughput     634.48 images/second |  Time   80.169 (s)
```

### Test on CIFAR-10
```
$ python test.py  --model models/model_CIFAR-10_029.npz
Loaded model for CIFAR-10 from  models/model_CIFAR-10_029.npz
Starting testing ...
....
Test accuracy: 0.8609225153923035
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
Epoch    0: Loss 3.61994 | Train accuracy  0.231 | Test accuracy  0.227 | Throughput     630.51 images/second |  Time   80.649 (s)
Epoch    1: Loss 2.84944 | Train accuracy  0.343 | Test accuracy  0.330 | Throughput     629.99 images/second |  Time   80.710 (s)
Epoch    2: Loss 2.44501 | Train accuracy  0.416 | Test accuracy  0.391 | Throughput     629.51 images/second |  Time   80.776 (s)
Epoch    3: Loss 2.20644 | Train accuracy  0.452 | Test accuracy  0.414 | Throughput     629.48 images/second |  Time   80.780 (s)
Epoch    4: Loss 2.03648 | Train accuracy  0.485 | Test accuracy  0.434 | Throughput     630.24 images/second |  Time   80.688 (s)
Epoch    5: Loss 1.91681 | Train accuracy  0.527 | Test accuracy  0.461 | Throughput     628.10 images/second |  Time   81.013 (s)
Epoch    6: Loss 1.81412 | Train accuracy  0.554 | Test accuracy  0.480 | Throughput     630.60 images/second |  Time   80.623 (s)
Epoch    7: Loss 1.72625 | Train accuracy  0.570 | Test accuracy  0.489 | Throughput     628.59 images/second |  Time   80.933 (s)
Epoch    8: Loss 1.65872 | Train accuracy  0.595 | Test accuracy  0.505 | Throughput     630.23 images/second |  Time   80.662 (s)
Epoch    9: Loss 1.59058 | Train accuracy  0.615 | Test accuracy  0.517 | Throughput     629.96 images/second |  Time   80.706 (s)
Epoch   10: Loss 1.53066 | Train accuracy  0.633 | Test accuracy  0.525 | Throughput     630.41 images/second |  Time   80.654 (s)
Epoch   11: Loss 1.47897 | Train accuracy  0.641 | Test accuracy  0.525 | Throughput     630.73 images/second |  Time   80.610 (s)
Epoch   12: Loss 1.42719 | Train accuracy  0.651 | Test accuracy  0.531 | Throughput     629.74 images/second |  Time   80.757 (s)
Epoch   13: Loss 1.38812 | Train accuracy  0.672 | Test accuracy  0.546 | Throughput     630.91 images/second |  Time   80.603 (s)
Epoch   14: Loss 1.35178 | Train accuracy  0.686 | Test accuracy  0.550 | Throughput     629.83 images/second |  Time   80.770 (s)
Epoch   15: Loss 1.30563 | Train accuracy  0.693 | Test accuracy  0.549 | Throughput     630.28 images/second |  Time   80.694 (s)
Epoch   16: Loss 1.27199 | Train accuracy  0.700 | Test accuracy  0.551 | Throughput     626.67 images/second |  Time   81.365 (s)
Epoch   17: Loss 1.24004 | Train accuracy  0.714 | Test accuracy  0.558 | Throughput     630.46 images/second |  Time   80.648 (s)
Epoch   18: Loss 1.21495 | Train accuracy  0.717 | Test accuracy  0.552 | Throughput     630.21 images/second |  Time   80.708 (s)
Epoch   19: Loss 1.17919 | Train accuracy  0.727 | Test accuracy  0.559 | Throughput     630.45 images/second |  Time   80.677 (s)
Epoch   20: Loss 1.15232 | Train accuracy  0.737 | Test accuracy  0.563 | Throughput     630.60 images/second |  Time   80.664 (s)
Epoch   21: Loss 1.12255 | Train accuracy  0.743 | Test accuracy  0.560 | Throughput     630.03 images/second |  Time   80.747 (s)
Epoch   22: Loss 1.10917 | Train accuracy  0.755 | Test accuracy  0.567 | Throughput     628.68 images/second |  Time   80.960 (s)
Epoch   23: Loss 1.07993 | Train accuracy  0.757 | Test accuracy  0.565 | Throughput     630.61 images/second |  Time   80.651 (s)
Epoch   24: Loss 1.06278 | Train accuracy  0.758 | Test accuracy  0.562 | Throughput     630.46 images/second |  Time   80.665 (s)
Epoch   25: Loss 1.03226 | Train accuracy  0.775 | Test accuracy  0.569 | Throughput     631.28 images/second |  Time   80.529 (s)
Epoch   26: Loss 1.01670 | Train accuracy  0.773 | Test accuracy  0.567 | Throughput     631.22 images/second |  Time   80.580 (s)
Epoch   27: Loss 0.99785 | Train accuracy  0.778 | Test accuracy  0.563 | Throughput     630.53 images/second |  Time   80.682 (s)
Epoch   28: Loss 0.97497 | Train accuracy  0.784 | Test accuracy  0.572 | Throughput     630.25 images/second |  Time   80.695 (s)
Epoch   29: Loss 0.95554 | Train accuracy  0.795 | Test accuracy  0.573 | Throughput     630.61 images/second |  Time   80.644 (s)

```

### Test on CIFAR-100
```
$ python test.py  --model models/model_CIFAR-100_029.npz --dataset=CIFAR-100
Loaded model for CIFAR-100 from models/model_CIFAR-100_029.npz
Starting testing ...
....
Test accuracy: 0.5730830430984497
```
