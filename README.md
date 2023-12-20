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
  Number of epochs: 30
  Learning rate: 0.0003
  Dataset: CIFAR-10
Number of trainable params: 0.3570 M
Starting training ...
Epoch    0: Loss 1.60208, Train accuracy 0.506, Test accuracy 0.508, Throughput 2181.64 images/second,  Time 23.869 (s)
Epoch    1: Loss 1.29547, Train accuracy 0.555, Test accuracy 0.544, Throughput 2221.90 images/second,  Time 23.263 (s)
Epoch    2: Loss 1.17564, Train accuracy 0.590, Test accuracy 0.574, Throughput 2137.24 images/second,  Time 24.200 (s)
Epoch    3: Loss 1.10677, Train accuracy 0.610, Test accuracy 0.590, Throughput 2155.42 images/second,  Time 23.933 (s)
Epoch    4: Loss 1.05182, Train accuracy 0.630, Test accuracy 0.615, Throughput 2214.60 images/second,  Time 23.335 (s)
Epoch    5: Loss 1.01217, Train accuracy 0.645, Test accuracy 0.625, Throughput 2206.26 images/second,  Time 23.405 (s)
Epoch    6: Loss 0.97722, Train accuracy 0.661, Test accuracy 0.642, Throughput 2166.95 images/second,  Time 23.816 (s)
Epoch    7: Loss 0.95499, Train accuracy 0.671, Test accuracy 0.642, Throughput 2154.87 images/second,  Time 23.938 (s)
Epoch    8: Loss 0.92478, Train accuracy 0.679, Test accuracy 0.649, Throughput 2151.28 images/second,  Time 23.982 (s)
Epoch    9: Loss 0.90832, Train accuracy 0.687, Test accuracy 0.659, Throughput 2161.54 images/second,  Time 23.865 (s)
Epoch   10: Loss 0.88897, Train accuracy 0.697, Test accuracy 0.662, Throughput 2157.17 images/second,  Time 23.912 (s)
Epoch   11: Loss 0.86823, Train accuracy 0.698, Test accuracy 0.661, Throughput 2157.35 images/second,  Time 23.912 (s)
Epoch   12: Loss 0.85403, Train accuracy 0.705, Test accuracy 0.671, Throughput 2160.81 images/second,  Time 23.900 (s)
Epoch   13: Loss 0.83622, Train accuracy 0.717, Test accuracy 0.675, Throughput 2172.26 images/second,  Time 23.888 (s)
Epoch   14: Loss 0.82264, Train accuracy 0.714, Test accuracy 0.672, Throughput 2153.67 images/second,  Time 24.621 (s)
Epoch   15: Loss 0.81252, Train accuracy 0.719, Test accuracy 0.684, Throughput 2179.55 images/second,  Time 23.913 (s)
Epoch   16: Loss 0.79851, Train accuracy 0.722, Test accuracy 0.685, Throughput 2227.93 images/second,  Time 23.236 (s)
Epoch   17: Loss 0.78523, Train accuracy 0.726, Test accuracy 0.682, Throughput 2210.65 images/second,  Time 23.502 (s)
Epoch   18: Loss 0.77848, Train accuracy 0.733, Test accuracy 0.684, Throughput 2171.63 images/second,  Time 23.892 (s)
Epoch   19: Loss 0.76383, Train accuracy 0.740, Test accuracy 0.689, Throughput 2223.80 images/second,  Time 23.218 (s)
Epoch   20: Loss 0.74815, Train accuracy 0.737, Test accuracy 0.691, Throughput 2192.71 images/second,  Time 23.554 (s)
Epoch   21: Loss 0.74346, Train accuracy 0.742, Test accuracy 0.695, Throughput 2234.65 images/second,  Time 23.123 (s)
Epoch   22: Loss 0.73211, Train accuracy 0.748, Test accuracy 0.693, Throughput 2231.08 images/second,  Time 23.110 (s)
Epoch   23: Loss 0.72311, Train accuracy 0.754, Test accuracy 0.699, Throughput 2219.91 images/second,  Time 23.246 (s)
Epoch   24: Loss 0.71656, Train accuracy 0.757, Test accuracy 0.704, Throughput 2151.00 images/second,  Time 24.292 (s)
Epoch   25: Loss 0.70498, Train accuracy 0.754, Test accuracy 0.700, Throughput 2181.98 images/second,  Time 23.884 (s)
Epoch   26: Loss 0.69597, Train accuracy 0.763, Test accuracy 0.712, Throughput 2151.17 images/second,  Time 24.523 (s)
Epoch   27: Loss 0.68542, Train accuracy 0.765, Test accuracy 0.709, Throughput 2139.84 images/second,  Time 24.400 (s)
Epoch   28: Loss 0.68440, Train accuracy 0.760, Test accuracy 0.702, Throughput 2165.59 images/second,  Time 23.978 (s)
Epoch   29: Loss 0.67292, Train accuracy 0.768, Test accuracy 0.709, Throughput 2124.82 images/second,  Time 24.647 (s)
```

### Test on CIFAR-10
```
$ python test.py  --model model_CIFAR-10_029.npz
Loaded model for CIFAR-10 from model_CIFAR-10_029.npz
Starting testing ...
....
Test accuracy: 0.7135583162307739
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
  Batch size: 32
  Number of epochs: 30
  Learning rate: 0.0003
  Dataset: CIFAR-100
Number of trainable params: 0.3686 M
Starting training ...
Epoch    0: Loss 4.12434, Train accuracy 0.126, Test accuracy 0.130, Throughput 2180.97 images/second,  Time 23.869 (s)
Epoch    1: Loss 3.51614, Train accuracy 0.184, Test accuracy 0.188, Throughput 2168.28 images/second,  Time 24.193 (s)
Epoch    2: Loss 3.27255, Train accuracy 0.217, Test accuracy 0.213, Throughput 2245.28 images/second,  Time 23.029 (s)
Epoch    3: Loss 3.10479, Train accuracy 0.245, Test accuracy 0.236, Throughput 2218.58 images/second,  Time 23.396 (s)
Epoch    4: Loss 2.97430, Train accuracy 0.273, Test accuracy 0.259, Throughput 2249.06 images/second,  Time 22.966 (s)
Epoch    5: Loss 2.86115, Train accuracy 0.290, Test accuracy 0.274, Throughput 2261.20 images/second,  Time 22.836 (s)
Epoch    6: Loss 2.77361, Train accuracy 0.305, Test accuracy 0.288, Throughput 2241.04 images/second,  Time 23.113 (s)
Epoch    7: Loss 2.70539, Train accuracy 0.322, Test accuracy 0.302, Throughput 2243.99 images/second,  Time 23.016 (s)
Epoch    8: Loss 2.64402, Train accuracy 0.331, Test accuracy 0.308, Throughput 2228.93 images/second,  Time 23.155 (s)
Epoch    9: Loss 2.59885, Train accuracy 0.342, Test accuracy 0.316, Throughput 2183.27 images/second,  Time 23.650 (s)
Epoch   10: Loss 2.55265, Train accuracy 0.349, Test accuracy 0.322, Throughput 2254.22 images/second,  Time 22.951 (s)
Epoch   11: Loss 2.51577, Train accuracy 0.355, Test accuracy 0.327, Throughput 2235.19 images/second,  Time 23.068 (s)
Epoch   12: Loss 2.48369, Train accuracy 0.364, Test accuracy 0.333, Throughput 2245.48 images/second,  Time 23.077 (s)
Epoch   13: Loss 2.45776, Train accuracy 0.368, Test accuracy 0.336, Throughput 2232.69 images/second,  Time 23.190 (s)
Epoch   14: Loss 2.43267, Train accuracy 0.378, Test accuracy 0.342, Throughput 2215.54 images/second,  Time 23.423 (s)
Epoch   15: Loss 2.40156, Train accuracy 0.380, Test accuracy 0.346, Throughput 2215.91 images/second,  Time 23.491 (s)
Epoch   16: Loss 2.37948, Train accuracy 0.384, Test accuracy 0.349, Throughput 2269.62 images/second,  Time 22.901 (s)
Epoch   17: Loss 2.35467, Train accuracy 0.391, Test accuracy 0.350, Throughput 2237.00 images/second,  Time 23.212 (s)
Epoch   18: Loss 2.33184, Train accuracy 0.398, Test accuracy 0.360, Throughput 2229.27 images/second,  Time 23.287 (s)
Epoch   19: Loss 2.31328, Train accuracy 0.403, Test accuracy 0.363, Throughput 2234.96 images/second,  Time 23.222 (s)
Epoch   20: Loss 2.30113, Train accuracy 0.407, Test accuracy 0.365, Throughput 2291.16 images/second,  Time 22.509 (s)
Epoch   21: Loss 2.27870, Train accuracy 0.406, Test accuracy 0.368, Throughput 2275.13 images/second,  Time 22.876 (s)
Epoch   22: Loss 2.26159, Train accuracy 0.410, Test accuracy 0.362, Throughput 2216.12 images/second,  Time 23.810 (s)
Epoch   23: Loss 2.24069, Train accuracy 0.416, Test accuracy 0.365, Throughput 2254.69 images/second,  Time 23.009 (s)
Epoch   24: Loss 2.22750, Train accuracy 0.414, Test accuracy 0.363, Throughput 2275.04 images/second,  Time 22.806 (s)
Epoch   25: Loss 2.20687, Train accuracy 0.416, Test accuracy 0.375, Throughput 2240.65 images/second,  Time 23.291 (s)
Epoch   26: Loss 2.19123, Train accuracy 0.429, Test accuracy 0.376, Throughput 2251.34 images/second,  Time 23.037 (s)
Epoch   27: Loss 2.17797, Train accuracy 0.424, Test accuracy 0.374, Throughput 2285.82 images/second,  Time 22.668 (s)
Epoch   28: Loss 2.16042, Train accuracy 0.431, Test accuracy 0.376, Throughput 2254.15 images/second,  Time 23.067 (s)
Epoch   29: Loss 2.14581, Train accuracy 0.429, Test accuracy 0.376, Throughput 2181.88 images/second,  Time 24.092 (s)

```

### Test on CIFAR-100
```
$ python test.py  --model model_CIFAR-100_029.npz --dataset=CIFAR-100
Loaded model for CIFAR-100 from model_CIFAR-100_029.npz
Starting testing ...
....
Test accuracy: 0.37799519300460815
```
