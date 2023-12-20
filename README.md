# mlx-cifar-10-cnn
Small CIFAR-10 / CIFAR-100 CNN implementation in Apple mlx. Used to try-out the first public version of https://github.com/ml-explore/mlx. It heavily borrows from the CIFAR example in https://github.com/ml-explore/mlx-examples

# Requirements
* Machine with Apple SoC (M1/M2/M3)
* mlx (https://github.com/ml-explore/mlx), tested with version 0.0.5.
* mlx-data (https://github.com/ml-explore/mlx-data)

# Limitations
* mlx does not yet include pooling layers, see https://github.com/ml-explore/mlx/issues/25

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
`python train.py`

This uses a first gen 16GB Macbook Pro M1. 

```
$ python train.py
Options: 
  Device: GPU
  Seed: 0
  Batch size: 32
  Number of epochs: 15
  Learning rate: 0.0003
  Dataset: CIFAR-10
Number of trainable params: 0.3570 M
Starting training ...
Epoch    0: Loss 1.59981, Train accuracy 0.507, Test accuracy 0.504, Throughput 2192.71 images/second,  Time 23.945 (s)
Epoch    1: Loss 1.29725, Train accuracy 0.556, Test accuracy 0.551, Throughput 2200.00 images/second,  Time 23.552 (s)
Epoch    2: Loss 1.17632, Train accuracy 0.596, Test accuracy 0.581, Throughput 2195.01 images/second,  Time 23.554 (s)
Epoch    3: Loss 1.10699, Train accuracy 0.617, Test accuracy 0.596, Throughput 2191.13 images/second,  Time 23.911 (s)
Epoch    4: Loss 1.05001, Train accuracy 0.635, Test accuracy 0.618, Throughput 2215.69 images/second,  Time 23.296 (s)
Epoch    5: Loss 1.01047, Train accuracy 0.644, Test accuracy 0.625, Throughput 2240.35 images/second,  Time 23.231 (s)
Epoch    6: Loss 0.97688, Train accuracy 0.658, Test accuracy 0.633, Throughput 2218.22 images/second,  Time 23.470 (s)
Epoch    7: Loss 0.95348, Train accuracy 0.672, Test accuracy 0.638, Throughput 2243.34 images/second,  Time 23.060 (s)
Epoch    8: Loss 0.92712, Train accuracy 0.682, Test accuracy 0.648, Throughput 2187.42 images/second,  Time 23.852 (s)
Epoch    9: Loss 0.90905, Train accuracy 0.688, Test accuracy 0.658, Throughput 2167.77 images/second,  Time 23.970 (s)
Epoch   10: Loss 0.88663, Train accuracy 0.695, Test accuracy 0.658, Throughput 2163.26 images/second,  Time 24.155 (s)
Epoch   11: Loss 0.86575, Train accuracy 0.701, Test accuracy 0.665, Throughput 2232.42 images/second,  Time 23.273 (s)
Epoch   12: Loss 0.84957, Train accuracy 0.710, Test accuracy 0.675, Throughput 2215.21 images/second,  Time 23.420 (s)
Epoch   13: Loss 0.83253, Train accuracy 0.717, Test accuracy 0.676, Throughput 2220.96 images/second,  Time 23.333 (s)
Epoch   14: Loss 0.81876, Train accuracy 0.716, Test accuracy 0.676, Throughput 2208.17 images/second,  Time 23.504 (s)
```

### Test on CIFAR-10
```
$ python test.py  --model model_CIFAR-10_014.npz
Loaded model for CIFAR-10 from model_CIFAR-10_014.npz
Starting testing ...
....
Test accuracy: 0.6774161458015442
```

## CIFAR-100

### Train on CIFAR-100
`python train.py --dataset CIFAR-100`

This uses a first gen 16GB Macbook Pro M1. 

```
$ python train.py
Options: 
  Device: GPU
  Seed: 0
  Batch size: 32
  Number of epochs: 15
  Learning rate: 0.0003
  Dataset: CIFAR-100
Number of trainable params: 0.3686 M
Starting training ...
Epoch    0: Loss 4.14763, Train accuracy 0.121, Test accuracy 0.123, Throughput 2194.11 images/second,  Time 23.534 (s)
Epoch    1: Loss 3.52998, Train accuracy 0.183, Test accuracy 0.186, Throughput 2178.48 images/second,  Time 23.810 (s)
Epoch    2: Loss 3.27450, Train accuracy 0.221, Test accuracy 0.215, Throughput 2232.84 images/second,  Time 23.312 (s)
Epoch    3: Loss 3.10801, Train accuracy 0.245, Test accuracy 0.239, Throughput 2221.58 images/second,  Time 23.256 (s)
Epoch    4: Loss 2.98208, Train accuracy 0.268, Test accuracy 0.261, Throughput 2200.31 images/second,  Time 23.567 (s)
Epoch    5: Loss 2.87788, Train accuracy 0.287, Test accuracy 0.276, Throughput 2198.29 images/second,  Time 23.649 (s)
Epoch    6: Loss 2.79474, Train accuracy 0.301, Test accuracy 0.285, Throughput 2177.19 images/second,  Time 23.818 (s)
Epoch    7: Loss 2.72837, Train accuracy 0.317, Test accuracy 0.302, Throughput 2190.70 images/second,  Time 23.691 (s)
Epoch    8: Loss 2.67476, Train accuracy 0.329, Test accuracy 0.314, Throughput 2217.20 images/second,  Time 23.449 (s)
Epoch    9: Loss 2.62040, Train accuracy 0.339, Test accuracy 0.318, Throughput 2255.54 images/second,  Time 23.162 (s)
Epoch   10: Loss 2.57351, Train accuracy 0.349, Test accuracy 0.326, Throughput 2054.92 images/second,  Time 26.352 (s)
Epoch   11: Loss 2.53486, Train accuracy 0.349, Test accuracy 0.325, Throughput 2283.01 images/second,  Time 22.684 (s)
Epoch   12: Loss 2.50743, Train accuracy 0.363, Test accuracy 0.333, Throughput 2296.87 images/second,  Time 22.572 (s)
Epoch   13: Loss 2.47160, Train accuracy 0.367, Test accuracy 0.340, Throughput 2262.03 images/second,  Time 22.868 (s)
Epoch   14: Loss 2.44610, Train accuracy 0.370, Test accuracy 0.338, Throughput 2221.29 images/second,  Time 23.362 (s)
```

### Test on CIFAR-100
```
$ python test.py  --model model_CIFAR-100_014.npz  --dataset=CIFAR-100
Loaded model for CIFAR-100 from model_CIFAR-100_014.npz
Starting testing ...
....
Test accuracy: 0.33206868171691895
```
