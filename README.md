# mlx-cifar-10-cnn
Small CIFAR-10 CNN implementation in Apple mlx. Used to try-out the first public version of https://github.com/ml-explore/mlx. It heavily borrows from the CIFAR example in https://github.com/ml-explore/mlx-examples

# Requirements
* Machine with Apple SoC (M1/M2/M3)
* mlx (https://github.com/ml-explore/mlx), tested with version 0.0.5.
* mlx-data (https://github.com/ml-explore/mlx-data)

# Limitations
* mlx does not yet include pooling layers, see https://github.com/ml-explore/mlx/issues/25

# Train
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
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz 162.6MiB (9.4MiB/s) 
Number of trainable params: 0.3570 M 
Starting training ...
Epoch    0: Loss 1.60110, Train accuracy 0.507, Test accuracy 0.507, Throughput 2237.00 images/second,  Time 23.205 (s)
Epoch    1: Loss 1.29622, Train accuracy 0.554, Test accuracy 0.548, Throughput 2220.12 images/second,  Time 23.258 (s)
Epoch    2: Loss 1.17690, Train accuracy 0.591, Test accuracy 0.573, Throughput 2195.28 images/second,  Time 23.869 (s)
Epoch    3: Loss 1.10781, Train accuracy 0.615, Test accuracy 0.603, Throughput 2225.02 images/second,  Time 23.269 (s)
Epoch    4: Loss 1.05267, Train accuracy 0.638, Test accuracy 0.619, Throughput 2189.69 images/second,  Time 23.625 (s)
Epoch    5: Loss 1.01267, Train accuracy 0.653, Test accuracy 0.630, Throughput 2168.93 images/second,  Time 23.842 (s)
Epoch    6: Loss 0.97861, Train accuracy 0.660, Test accuracy 0.638, Throughput 2230.18 images/second,  Time 23.264 (s)
Epoch    7: Loss 0.95356, Train accuracy 0.674, Test accuracy 0.649, Throughput 2251.04 images/second,  Time 23.060 (s)
Epoch    8: Loss 0.92774, Train accuracy 0.686, Test accuracy 0.655, Throughput 2183.01 images/second,  Time 23.734 (s)
Epoch    9: Loss 0.90558, Train accuracy 0.688, Test accuracy 0.659, Throughput 2217.17 images/second,  Time 23.674 (s)
Epoch   10: Loss 0.88850, Train accuracy 0.702, Test accuracy 0.667, Throughput 2207.08 images/second,  Time 23.689 (s)
Epoch   11: Loss 0.86691, Train accuracy 0.698, Test accuracy 0.662, Throughput 2160.21 images/second,  Time 24.074 (s)
Epoch   12: Loss 0.85118, Train accuracy 0.705, Test accuracy 0.667, Throughput 2206.40 images/second,  Time 23.481 (s)
Epoch   13: Loss 0.83440, Train accuracy 0.716, Test accuracy 0.677, Throughput 2246.39 images/second,  Time 23.018 (s)
Epoch   14: Loss 0.82287, Train accuracy 0.715, Test accuracy 0.672, Throughput 2174.63 images/second,  Time 23.819 (s)
```

# Test
```
$ python test.py  --model model_014.npz
Loaded model from model_014.npz
Starting testing ...
....
Test accuracy: 0.6712260246276855
```