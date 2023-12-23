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
Epoch    0: Loss 1.60136, Train accuracy 0.532, Test accuracy 0.527, Throughput 2159.17 images/second,  Time 24.213 (s)
Epoch    1: Loss 1.29727, Train accuracy 0.584, Test accuracy 0.575, Throughput 2191.67 images/second,  Time 23.734 (s)
Epoch    2: Loss 1.17798, Train accuracy 0.624, Test accuracy 0.610, Throughput 2188.16 images/second,  Time 23.619 (s)
Epoch    3: Loss 1.10689, Train accuracy 0.652, Test accuracy 0.636, Throughput 2152.90 images/second,  Time 24.072 (s)
Epoch    4: Loss 1.05353, Train accuracy 0.672, Test accuracy 0.651, Throughput 2201.85 images/second,  Time 23.656 (s)
Epoch    5: Loss 1.01629, Train accuracy 0.684, Test accuracy 0.662, Throughput 2222.74 images/second,  Time 23.332 (s)
Epoch    6: Loss 0.98483, Train accuracy 0.697, Test accuracy 0.672, Throughput 2189.81 images/second,  Time 23.678 (s)
Epoch    7: Loss 0.95860, Train accuracy 0.710, Test accuracy 0.682, Throughput 2245.45 images/second,  Time 23.071 (s)
Epoch    8: Loss 0.92920, Train accuracy 0.721, Test accuracy 0.687, Throughput 2213.53 images/second,  Time 23.390 (s)
Epoch    9: Loss 0.91159, Train accuracy 0.732, Test accuracy 0.697, Throughput 2171.55 images/second,  Time 23.929 (s)
Epoch   10: Loss 0.89003, Train accuracy 0.734, Test accuracy 0.699, Throughput 2104.64 images/second,  Time 25.381 (s)
Epoch   11: Loss 0.87531, Train accuracy 0.744, Test accuracy 0.705, Throughput 2218.32 images/second,  Time 23.320 (s)
Epoch   12: Loss 0.86116, Train accuracy 0.751, Test accuracy 0.711, Throughput 2202.79 images/second,  Time 23.846 (s)
Epoch   13: Loss 0.84236, Train accuracy 0.750, Test accuracy 0.709, Throughput 2233.09 images/second,  Time 23.118 (s)
Epoch   14: Loss 0.83182, Train accuracy 0.770, Test accuracy 0.728, Throughput 2213.79 images/second,  Time 23.592 (s)
Epoch   15: Loss 0.81768, Train accuracy 0.769, Test accuracy 0.723, Throughput 2246.00 images/second,  Time 22.983 (s)
Epoch   16: Loss 0.79888, Train accuracy 0.778, Test accuracy 0.734, Throughput 2204.45 images/second,  Time 23.421 (s)
Epoch   17: Loss 0.78891, Train accuracy 0.777, Test accuracy 0.728, Throughput 2222.98 images/second,  Time 23.220 (s)
Epoch   18: Loss 0.77542, Train accuracy 0.782, Test accuracy 0.726, Throughput 2263.93 images/second,  Time 22.827 (s)
Epoch   19: Loss 0.76866, Train accuracy 0.787, Test accuracy 0.737, Throughput 2263.56 images/second,  Time 22.828 (s)
Epoch   20: Loss 0.75714, Train accuracy 0.795, Test accuracy 0.741, Throughput 2234.76 images/second,  Time 23.146 (s)
Epoch   21: Loss 0.74696, Train accuracy 0.802, Test accuracy 0.743, Throughput 2201.25 images/second,  Time 23.626 (s)
Epoch   22: Loss 0.73565, Train accuracy 0.801, Test accuracy 0.741, Throughput 2255.59 images/second,  Time 22.870 (s)
Epoch   23: Loss 0.72661, Train accuracy 0.803, Test accuracy 0.742, Throughput 2175.99 images/second,  Time 23.837 (s)
Epoch   24: Loss 0.71646, Train accuracy 0.807, Test accuracy 0.745, Throughput 2206.44 images/second,  Time 23.704 (s)
Epoch   25: Loss 0.70552, Train accuracy 0.813, Test accuracy 0.752, Throughput 2232.99 images/second,  Time 23.184 (s)
Epoch   26: Loss 0.70079, Train accuracy 0.822, Test accuracy 0.759, Throughput 2232.84 images/second,  Time 23.106 (s)
Epoch   27: Loss 0.69554, Train accuracy 0.819, Test accuracy 0.752, Throughput 2261.45 images/second,  Time 22.931 (s)
Epoch   28: Loss 0.68483, Train accuracy 0.819, Test accuracy 0.750, Throughput 2210.85 images/second,  Time 23.426 (s)
Epoch   29: Loss 0.67732, Train accuracy 0.827, Test accuracy 0.757, Throughput 2216.58 images/second,  Time 23.442 (s)
```

### Test on CIFAR-10
```
$ python test.py  --model model_CIFAR-10_029.npz
Loaded model for CIFAR-10 from model_CIFAR-10_029.npz
Starting testing ...
....
Test accuracy: 0.7568889856338501
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
Epoch    0: Loss 4.10041, Train accuracy 0.139, Test accuracy 0.145, Throughput 2206.30 images/second,  Time 23.540 (s)
Epoch    1: Loss 3.47617, Train accuracy 0.204, Test accuracy 0.202, Throughput 2251.84 images/second,  Time 23.056 (s)
Epoch    2: Loss 3.23598, Train accuracy 0.245, Test accuracy 0.237, Throughput 2237.80 images/second,  Time 23.109 (s)
Epoch    3: Loss 3.08060, Train accuracy 0.278, Test accuracy 0.271, Throughput 2202.76 images/second,  Time 23.455 (s)
Epoch    4: Loss 2.95265, Train accuracy 0.305, Test accuracy 0.294, Throughput 2206.85 images/second,  Time 23.451 (s)
Epoch    5: Loss 2.84587, Train accuracy 0.328, Test accuracy 0.313, Throughput 2200.18 images/second,  Time 23.470 (s)
Epoch    6: Loss 2.75509, Train accuracy 0.349, Test accuracy 0.325, Throughput 2177.21 images/second,  Time 23.894 (s)
Epoch    7: Loss 2.68642, Train accuracy 0.359, Test accuracy 0.338, Throughput 2200.46 images/second,  Time 23.737 (s)
Epoch    8: Loss 2.62358, Train accuracy 0.377, Test accuracy 0.348, Throughput 2163.85 images/second,  Time 24.096 (s)
Epoch    9: Loss 2.57836, Train accuracy 0.391, Test accuracy 0.359, Throughput 2099.72 images/second,  Time 25.320 (s)
Epoch   10: Loss 2.52761, Train accuracy 0.400, Test accuracy 0.366, Throughput 2170.32 images/second,  Time 23.785 (s)
Epoch   11: Loss 2.48650, Train accuracy 0.408, Test accuracy 0.373, Throughput 2149.86 images/second,  Time 24.500 (s)
Epoch   12: Loss 2.46098, Train accuracy 0.414, Test accuracy 0.373, Throughput 2231.63 images/second,  Time 23.311 (s)
Epoch   13: Loss 2.42272, Train accuracy 0.419, Test accuracy 0.378, Throughput 2183.97 images/second,  Time 23.729 (s)
Epoch   14: Loss 2.39206, Train accuracy 0.431, Test accuracy 0.384, Throughput 2176.74 images/second,  Time 23.866 (s)
Epoch   15: Loss 2.36049, Train accuracy 0.431, Test accuracy 0.386, Throughput 2206.87 images/second,  Time 23.474 (s)
Epoch   16: Loss 2.33826, Train accuracy 0.443, Test accuracy 0.394, Throughput 2183.50 images/second,  Time 34.916 (s)
Epoch   17: Loss 2.32027, Train accuracy 0.452, Test accuracy 0.398, Throughput 2163.15 images/second,  Time 24.013 (s)
Epoch   18: Loss 2.28781, Train accuracy 0.457, Test accuracy 0.400, Throughput 2157.19 images/second,  Time 24.042 (s)
Epoch   19: Loss 2.27234, Train accuracy 0.458, Test accuracy 0.403, Throughput 2153.31 images/second,  Time 24.135 (s)
Epoch   20: Loss 2.24914, Train accuracy 0.469, Test accuracy 0.408, Throughput 2184.73 images/second,  Time 23.672 (s)
Epoch   21: Loss 2.22976, Train accuracy 0.472, Test accuracy 0.415, Throughput 2177.69 images/second,  Time 23.802 (s)
Epoch   22: Loss 2.21521, Train accuracy 0.475, Test accuracy 0.414, Throughput 2155.53 images/second,  Time 23.982 (s)
Epoch   23: Loss 2.19268, Train accuracy 0.486, Test accuracy 0.419, Throughput 2160.73 images/second,  Time 24.103 (s)
Epoch   24: Loss 2.17588, Train accuracy 0.493, Test accuracy 0.422, Throughput 2110.16 images/second,  Time 24.596 (s)
Epoch   25: Loss 2.15501, Train accuracy 0.498, Test accuracy 0.425, Throughput 1953.37 images/second,  Time 27.048 (s)
Epoch   26: Loss 2.14772, Train accuracy 0.497, Test accuracy 0.425, Throughput 2187.23 images/second,  Time 23.731 (s)
Epoch   27: Loss 2.12451, Train accuracy 0.505, Test accuracy 0.429, Throughput 2168.10 images/second,  Time 23.924 (s)
Epoch   28: Loss 2.10977, Train accuracy 0.508, Test accuracy 0.428, Throughput 2199.52 images/second,  Time 23.581 (s)
Epoch   29: Loss 2.09999, Train accuracy 0.511, Test accuracy 0.432, Throughput 2207.95 images/second,  Time 23.391 (s)

```

### Test on CIFAR-100
```
$ python test.py  --model model_CIFAR-100_029.npz --dataset=CIFAR-100
Loaded model for CIFAR-100 from model_CIFAR-100_029.npz
Starting testing ...
....
Test accuracy: 0.4320087730884552
```
