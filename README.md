# mlx-cifar-10-cnn
Small CIFAR-10 CNN implementation in Apple mlx. Used to try-out the first public version of https://github.com/ml-explore/mlx
It heavily borrows from the CIFAR example in https://github.com/ml-explore/mlx-examples

# Requirements
* Machine with Apple SoC (M1/M2/M3)
* mlx (https://github.com/ml-explore/mlx), tested with version 0.0.5. Install using instructions on https://github.com/ml-explore/mlx
* mlx-data (https://github.com/ml-explore/mlx-data)

# Limitations
* mlx does not yet include pooling layers, see https://github.com/ml-explore/mlx/issues/25

# Train
`python train.py`

This uses a first gen 16GB Macbook Pro M1. 

```
(mlx-m1-2023-12) $ python train.py
Options:
  Device: GPU
  Seed: 0
  Batch size: 64
  Number of epochs: 20
  Learning rate: 0.0005
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz 162.6MiB (9.4MiB/s)  
Starting training ...
Epoch    0: Loss 1.59012, Train accuracy 0.517, Test accuracy 0.515, Throughput 2064.38 images/second,  Time 24.960 (s)
Epoch    1: Loss 1.28264, Train accuracy 0.568, Test accuracy 0.555, Throughput 1858.34 images/second,  Time 27.714 (s)
Epoch    2: Loss 1.15834, Train accuracy 0.600, Test accuracy 0.577, Throughput 1835.21 images/second,  Time 28.119 (s)
Epoch    3: Loss 1.08163, Train accuracy 0.628, Test accuracy 0.599, Throughput 1886.10 images/second,  Time 27.431 (s)
Epoch    4: Loss 1.02606, Train accuracy 0.640, Test accuracy 0.612, Throughput 1829.63 images/second,  Time 28.562 (s)
Epoch    5: Loss 0.98446, Train accuracy 0.662, Test accuracy 0.629, Throughput 1873.30 images/second,  Time 27.566 (s)
Epoch    6: Loss 0.94463, Train accuracy 0.675, Test accuracy 0.635, Throughput 1891.81 images/second,  Time 27.382 (s)
Epoch    7: Loss 0.91536, Train accuracy 0.675, Test accuracy 0.627, Throughput 1901.21 images/second,  Time 27.366 (s)
Epoch    8: Loss 0.88837, Train accuracy 0.694, Test accuracy 0.650, Throughput 1875.42 images/second,  Time 27.911 (s)
Epoch    9: Loss 0.86175, Train accuracy 0.697, Test accuracy 0.644, Throughput 1860.57 images/second,  Time 28.070 (s)
Epoch   10: Loss 0.83467, Train accuracy 0.706, Test accuracy 0.649, Throughput 1891.46 images/second,  Time 27.286 (s)
Epoch   11: Loss 0.81078, Train accuracy 0.714, Test accuracy 0.650, Throughput 1867.83 images/second,  Time 27.600 (s)
Epoch   12: Loss 0.79714, Train accuracy 0.729, Test accuracy 0.663, Throughput 1938.69 images/second,  Time 26.697 (s)
Epoch   13: Loss 0.78267, Train accuracy 0.733, Test accuracy 0.662, Throughput 1844.71 images/second,  Time 28.192 (s)
Epoch   14: Loss 0.76080, Train accuracy 0.739, Test accuracy 0.662, Throughput 1845.13 images/second,  Time 27.991 (s)
```

# Test
```
(mlx-m1-2023-12) $ python test.py  --model model_014.npz
Loaded model from model_014.npz
Starting testing ...
....
Test accuracy: 0.6621328808784485
```