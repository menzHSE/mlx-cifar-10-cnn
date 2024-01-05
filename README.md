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
Epoch    0: Loss 1.36592 | Train accuracy  0.633 | Test accuracy  0.621 | Throughput     701.33 images/second |  Time   73.710 (s)
Epoch    1: Loss 1.00033 | Train accuracy  0.707 | Test accuracy  0.686 | Throughput     708.50 images/second |  Time   72.834 (s)
Epoch    2: Loss 0.84625 | Train accuracy  0.755 | Test accuracy  0.725 | Throughput     683.04 images/second |  Time   75.454 (s)
Epoch    3: Loss 0.74630 | Train accuracy  0.790 | Test accuracy  0.756 | Throughput     684.74 images/second |  Time   75.356 (s)
Epoch    4: Loss 0.67480 | Train accuracy  0.807 | Test accuracy  0.762 | Throughput     700.95 images/second |  Time   73.787 (s)
Epoch    5: Loss 0.62728 | Train accuracy  0.825 | Test accuracy  0.774 | Throughput     678.08 images/second |  Time   76.348 (s)
Epoch    6: Loss 0.58239 | Train accuracy  0.834 | Test accuracy  0.784 | Throughput     682.26 images/second |  Time   75.678 (s)
Epoch    7: Loss 0.54405 | Train accuracy  0.850 | Test accuracy  0.791 | Throughput     683.67 images/second |  Time   75.478 (s)
Epoch    8: Loss 0.51322 | Train accuracy  0.865 | Test accuracy  0.797 | Throughput     680.71 images/second |  Time   75.854 (s)
Epoch    9: Loss 0.48491 | Train accuracy  0.875 | Test accuracy  0.806 | Throughput     680.27 images/second |  Time   76.003 (s)
Epoch   10: Loss 0.46560 | Train accuracy  0.882 | Test accuracy  0.807 | Throughput     679.44 images/second |  Time   76.022 (s)
Epoch   11: Loss 0.43820 | Train accuracy  0.887 | Test accuracy  0.807 | Throughput     679.97 images/second |  Time   75.929 (s)
Epoch   12: Loss 0.42052 | Train accuracy  0.898 | Test accuracy  0.812 | Throughput     680.28 images/second |  Time   75.874 (s)
Epoch   13: Loss 0.40392 | Train accuracy  0.904 | Test accuracy  0.816 | Throughput     687.90 images/second |  Time   74.971 (s)
Epoch   14: Loss 0.38257 | Train accuracy  0.910 | Test accuracy  0.815 | Throughput     672.95 images/second |  Time   76.880 (s)
Epoch   15: Loss 0.36881 | Train accuracy  0.914 | Test accuracy  0.815 | Throughput     687.32 images/second |  Time   75.097 (s)
Epoch   16: Loss 0.35247 | Train accuracy  0.922 | Test accuracy  0.822 | Throughput     690.68 images/second |  Time   74.763 (s)
Epoch   17: Loss 0.33898 | Train accuracy  0.922 | Test accuracy  0.815 | Throughput     689.28 images/second |  Time   74.865 (s)
Epoch   18: Loss 0.32534 | Train accuracy  0.931 | Test accuracy  0.824 | Throughput     687.49 images/second |  Time   75.150 (s)
Epoch   19: Loss 0.31865 | Train accuracy  0.931 | Test accuracy  0.823 | Throughput     684.15 images/second |  Time   75.483 (s)
Epoch   20: Loss 0.30603 | Train accuracy  0.935 | Test accuracy  0.823 | Throughput     686.21 images/second |  Time   75.199 (s)
Epoch   21: Loss 0.29161 | Train accuracy  0.942 | Test accuracy  0.828 | Throughput     686.26 images/second |  Time   75.263 (s)
Epoch   22: Loss 0.28813 | Train accuracy  0.944 | Test accuracy  0.826 | Throughput     684.66 images/second |  Time   75.460 (s)
Epoch   23: Loss 0.27645 | Train accuracy  0.947 | Test accuracy  0.826 | Throughput     685.68 images/second |  Time   75.373 (s)
Epoch   24: Loss 0.26543 | Train accuracy  0.949 | Test accuracy  0.826 | Throughput     684.72 images/second |  Time   75.475 (s)
Epoch   25: Loss 0.25949 | Train accuracy  0.952 | Test accuracy  0.830 | Throughput     708.27 images/second |  Time   72.894 (s)
Epoch   26: Loss 0.25745 | Train accuracy  0.954 | Test accuracy  0.829 | Throughput     691.90 images/second |  Time   74.639 (s)
Epoch   27: Loss 0.24579 | Train accuracy  0.960 | Test accuracy  0.834 | Throughput     682.59 images/second |  Time   75.690 (s)
Epoch   28: Loss 0.23759 | Train accuracy  0.957 | Test accuracy  0.827 | Throughput     714.69 images/second |  Time   72.162 (s)
Epoch   29: Loss 0.22961 | Train accuracy  0.961 | Test accuracy  0.826 | Throughput     740.28 images/second |  Time   69.703 (s)
```

### Test on CIFAR-10
```
$ python test.py  --model models/model_CIFAR-10_029.npz
Loaded model for CIFAR-10 from  models/model_CIFAR-10_029.npz
Starting testing ...
....
Test accuracy: 0.8257787227630615
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
Epoch    0: Loss 3.62411 | Train accuracy  0.250 | Test accuracy  0.242 | Throughput     774.63 images/second |  Time   65.939 (s)
Epoch    1: Loss 2.90617 | Train accuracy  0.344 | Test accuracy  0.318 | Throughput     772.65 images/second |  Time   66.112 (s)
Epoch    2: Loss 2.55204 | Train accuracy  0.402 | Test accuracy  0.362 | Throughput     766.77 images/second |  Time   66.737 (s)
Epoch    3: Loss 2.33260 | Train accuracy  0.442 | Test accuracy  0.395 | Throughput     691.81 images/second |  Time   74.753 (s)
Epoch    4: Loss 2.17116 | Train accuracy  0.477 | Test accuracy  0.417 | Throughput     683.82 images/second |  Time   75.576 (s)
Epoch    5: Loss 2.03544 | Train accuracy  0.510 | Test accuracy  0.442 | Throughput     685.54 images/second |  Time   75.392 (s)
Epoch    6: Loss 1.93557 | Train accuracy  0.539 | Test accuracy  0.454 | Throughput     680.16 images/second |  Time   75.942 (s)
Epoch    7: Loss 1.85213 | Train accuracy  0.554 | Test accuracy  0.459 | Throughput     678.02 images/second |  Time   76.171 (s)
Epoch    8: Loss 1.76859 | Train accuracy  0.578 | Test accuracy  0.476 | Throughput     680.97 images/second |  Time   75.832 (s)
Epoch    9: Loss 1.71155 | Train accuracy  0.591 | Test accuracy  0.476 | Throughput     682.08 images/second |  Time   75.760 (s)
Epoch   10: Loss 1.65093 | Train accuracy  0.610 | Test accuracy  0.488 | Throughput     694.64 images/second |  Time   74.459 (s)
Epoch   11: Loss 1.59735 | Train accuracy  0.623 | Test accuracy  0.491 | Throughput     684.35 images/second |  Time   75.508 (s)
Epoch   12: Loss 1.54614 | Train accuracy  0.640 | Test accuracy  0.503 | Throughput     684.50 images/second |  Time   75.463 (s)
Epoch   13: Loss 1.49253 | Train accuracy  0.661 | Test accuracy  0.508 | Throughput     684.29 images/second |  Time   75.456 (s)
Epoch   14: Loss 1.45124 | Train accuracy  0.669 | Test accuracy  0.509 | Throughput     676.85 images/second |  Time   76.570 (s)
Epoch   15: Loss 1.41013 | Train accuracy  0.680 | Test accuracy  0.517 | Throughput     688.77 images/second |  Time   74.793 (s)
Epoch   16: Loss 1.37820 | Train accuracy  0.693 | Test accuracy  0.515 | Throughput     687.12 images/second |  Time   75.075 (s)
Epoch   17: Loss 1.33706 | Train accuracy  0.705 | Test accuracy  0.523 | Throughput     698.67 images/second |  Time   73.904 (s)
Epoch   18: Loss 1.29861 | Train accuracy  0.711 | Test accuracy  0.521 | Throughput     689.82 images/second |  Time   74.801 (s)
Epoch   19: Loss 1.27196 | Train accuracy  0.722 | Test accuracy  0.523 | Throughput     690.74 images/second |  Time   74.853 (s)
Epoch   20: Loss 1.24367 | Train accuracy  0.733 | Test accuracy  0.527 | Throughput     694.63 images/second |  Time   74.306 (s)
Epoch   21: Loss 1.21895 | Train accuracy  0.743 | Test accuracy  0.532 | Throughput     686.31 images/second |  Time   75.296 (s)
Epoch   22: Loss 1.18503 | Train accuracy  0.745 | Test accuracy  0.531 | Throughput     685.15 images/second |  Time   75.453 (s)
Epoch   23: Loss 1.16028 | Train accuracy  0.761 | Test accuracy  0.537 | Throughput     693.56 images/second |  Time   74.228 (s)
Epoch   24: Loss 1.13215 | Train accuracy  0.760 | Test accuracy  0.534 | Throughput     686.83 images/second |  Time   75.061 (s)
Epoch   25: Loss 1.11408 | Train accuracy  0.780 | Test accuracy  0.541 | Throughput     675.45 images/second |  Time   76.484 (s)
Epoch   26: Loss 1.08808 | Train accuracy  0.781 | Test accuracy  0.540 | Throughput     668.23 images/second |  Time   77.364 (s)
Epoch   27: Loss 1.07467 | Train accuracy  0.785 | Test accuracy  0.542 | Throughput     688.19 images/second |  Time   75.054 (s)
Epoch   28: Loss 1.05003 | Train accuracy  0.790 | Test accuracy  0.541 | Throughput     688.58 images/second |  Time   74.930 (s)
Epoch   29: Loss 1.03433 | Train accuracy  0.799 | Test accuracy  0.543 | Throughput     706.21 images/second |  Time   73.218 (s)

```

### Test on CIFAR-100
```
$ python test.py  --model models/model_CIFAR-100_029.npz --dataset=CIFAR-100
Loaded model for CIFAR-100 from models/model_CIFAR-100_029.npz
Starting testing ...
....
Test accuracy: 0.5409345030784607
```
