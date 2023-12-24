# mlx-cifar-10-cnn
Small CIFAR-10 / CIFAR-100 CNN implementation in Apple mlx. Used to try-out the first public version of https://github.com/ml-explore/mlx. It heavily borrows from the CIFAR example in https://github.com/ml-explore/mlx-examples

See https://github.com/menzHSE/torch-cifar-10-cnn for (more or less) the same model being trained using PyTorch.

# Requirements
* Machine with Apple SoC (M1/M2/M3)
* mlx (https://github.com/ml-explore/mlx), tested with version 0.0.6.
* mlx-data (https://github.com/ml-explore/mlx-data)

# Limitations
* mlx does not yet include pooling layers, see https://github.com/ml-explore/mlx/issues/25, so we use strided convolution width `stride=2` for subsampling
* mlx does not yet include batch norm, see https://github.com/ml-explore/mlx/pull/217, so we use layer norm instead
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
$ python train.py --dataset=CIFAR-10
Options: 
  Device: GPU
  Seed: 0
  Batch size: 128
  Number of epochs: 30
  Learning rate: 0.0003
  Dataset: CIFAR-10
Number of trainable params: 0.5515 M
Starting training ...
Epoch    0: Loss 1.83104, Train accuracy 0.466, Test accuracy 0.457, Throughput 939.36 images/second,  Time 55.034 (s)
Epoch    1: Loss 1.34206, Train accuracy 0.587, Test accuracy 0.571, Throughput 932.24 images/second,  Time 55.436 (s)
Epoch    2: Loss 1.14220, Train accuracy 0.644, Test accuracy 0.630, Throughput 943.30 images/second,  Time 54.975 (s)
Epoch    3: Loss 1.02513, Train accuracy 0.684, Test accuracy 0.671, Throughput 943.75 images/second,  Time 54.971 (s)
Epoch    4: Loss 0.95073, Train accuracy 0.701, Test accuracy 0.683, Throughput 943.57 images/second,  Time 54.991 (s)
Epoch    5: Loss 0.89586, Train accuracy 0.712, Test accuracy 0.692, Throughput 943.79 images/second,  Time 54.948 (s)
Epoch    6: Loss 0.85462, Train accuracy 0.742, Test accuracy 0.718, Throughput 939.29 images/second,  Time 55.207 (s)
Epoch    7: Loss 0.81586, Train accuracy 0.749, Test accuracy 0.719, Throughput 933.47 images/second,  Time 55.491 (s)
Epoch    8: Loss 0.78270, Train accuracy 0.762, Test accuracy 0.730, Throughput 937.94 images/second,  Time 55.273 (s)
Epoch    9: Loss 0.75561, Train accuracy 0.771, Test accuracy 0.739, Throughput 941.93 images/second,  Time 55.086 (s)
Epoch   10: Loss 0.72990, Train accuracy 0.788, Test accuracy 0.745, Throughput 940.34 images/second,  Time 55.165 (s)
Epoch   11: Loss 0.70371, Train accuracy 0.795, Test accuracy 0.757, Throughput 943.46 images/second,  Time 54.980 (s)
Epoch   12: Loss 0.68704, Train accuracy 0.797, Test accuracy 0.755, Throughput 946.19 images/second,  Time 54.800 (s)
Epoch   13: Loss 0.65954, Train accuracy 0.804, Test accuracy 0.762, Throughput 944.83 images/second,  Time 54.875 (s)
Epoch   14: Loss 0.64492, Train accuracy 0.815, Test accuracy 0.767, Throughput 940.90 images/second,  Time 55.149 (s)
Epoch   15: Loss 0.63009, Train accuracy 0.824, Test accuracy 0.774, Throughput 940.32 images/second,  Time 55.176 (s)
Epoch   16: Loss 0.61107, Train accuracy 0.828, Test accuracy 0.776, Throughput 947.16 images/second,  Time 54.728 (s)
Epoch   17: Loss 0.59497, Train accuracy 0.833, Test accuracy 0.781, Throughput 940.62 images/second,  Time 55.160 (s)
Epoch   18: Loss 0.58061, Train accuracy 0.834, Test accuracy 0.782, Throughput 943.18 images/second,  Time 55.006 (s)
Epoch   19: Loss 0.57207, Train accuracy 0.844, Test accuracy 0.787, Throughput 942.81 images/second,  Time 55.021 (s)
Epoch   20: Loss 0.55474, Train accuracy 0.850, Test accuracy 0.792, Throughput 942.20 images/second,  Time 55.051 (s)
Epoch   21: Loss 0.54379, Train accuracy 0.856, Test accuracy 0.793, Throughput 939.76 images/second,  Time 55.214 (s)
Epoch   22: Loss 0.53801, Train accuracy 0.849, Test accuracy 0.790, Throughput 933.88 images/second,  Time 55.551 (s)
Epoch   23: Loss 0.52973, Train accuracy 0.865, Test accuracy 0.803, Throughput 940.28 images/second,  Time 55.177 (s)
Epoch   24: Loss 0.51630, Train accuracy 0.868, Test accuracy 0.800, Throughput 941.58 images/second,  Time 55.090 (s)
Epoch   25: Loss 0.50128, Train accuracy 0.867, Test accuracy 0.798, Throughput 940.54 images/second,  Time 55.145 (s)
Epoch   26: Loss 0.49484, Train accuracy 0.873, Test accuracy 0.802, Throughput 940.03 images/second,  Time 55.187 (s)
Epoch   27: Loss 0.49035, Train accuracy 0.871, Test accuracy 0.804, Throughput 942.55 images/second,  Time 54.951 (s)
Epoch   28: Loss 0.48336, Train accuracy 0.881, Test accuracy 0.810, Throughput 938.92 images/second,  Time 55.260 (s)
Epoch   29: Loss 0.47195, Train accuracy 0.882, Test accuracy 0.809, Throughput 940.32 images/second,  Time 55.168 (s)
```

### Test on CIFAR-10
```
$ python test.py  --model model_CIFAR-10_029.npz
Loaded model for CIFAR-10 from model_CIFAR-10_029.npz
Starting testing ...
....
Test accuracy: 0.8083066940307617
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
