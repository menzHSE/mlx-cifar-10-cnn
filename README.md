# mlx-cifar-10-cnn
Small CIFAR-10 CNN implementation in Apple mlx. Used to try-out the first public version of https://github.com/ml-explore/mlx
It heavily borrows from the CIFAR example in https://github.com/ml-explore/mlx-examples

# Requirements
* Machine with Apple silicon (M1/M2/M3)
* mlx (tested with version 0.0.5). Install using instructions on https://github.com/ml-explore/mlx
* mlx-data

# Limitations
* mlx does not yet include pooling layers, see https://github.com/ml-explore/mlx/issues/25

# Run
`python train.py  --gpu --batchsize=64 --nepoch=20 --lr=7e-4`

This uses a first gen 16GB Macbook Pro M1. 

```
(mlx-m1-2023-12) $ python train.py  --gpu --batchsize=64 --nepoch=20 --lr=7e-4
Options: 
  GPU: True
  Seed: 0
  Batch size: 64
  Number of epochs: 20
  Learning rate: 0.0007
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz 162.6MiB (9.4MiB/s)  
Starting training ...
Epoch   0: Loss 1.80195, Train accuracy 0.430, Test accuracy 0.427, Throughput 663.59 images/second,  Time 78.182 (s)
Epoch   1: Loss 1.45143, Train accuracy 0.502, Test accuracy 0.495, Throughput 667.24 images/second,  Time 78.143 (s)
Epoch   2: Loss 1.32624, Train accuracy 0.544, Test accuracy 0.532, Throughput 683.33 images/second,  Time 75.819 (s)
Epoch   3: Loss 1.23480, Train accuracy 0.574, Test accuracy 0.558, Throughput 666.66 images/second,  Time 78.422 (s)
Epoch   4: Loss 1.16394, Train accuracy 0.595, Test accuracy 0.572, Throughput 663.24 images/second,  Time 78.737 (s)
Epoch   5: Loss 1.10512, Train accuracy 0.621, Test accuracy 0.599, Throughput 659.59 images/second,  Time 79.032 (s)
Epoch   6: Loss 1.06172, Train accuracy 0.625, Test accuracy 0.605, Throughput 663.62 images/second,  Time 78.614 (s)
Epoch   7: Loss 1.02378, Train accuracy 0.642, Test accuracy 0.619, Throughput 662.41 images/second,  Time 78.710 (s)
Epoch   8: Loss 0.98360, Train accuracy 0.660, Test accuracy 0.640, Throughput 658.85 images/second,  Time 79.157 (s)
Epoch   9: Loss 0.95453, Train accuracy 0.664, Test accuracy 0.648, Throughput 658.52 images/second,  Time 79.267 (s)
Epoch  10: Loss 0.92667, Train accuracy 0.680, Test accuracy 0.654, Throughput 660.18 images/second,  Time 79.001 (s)
Epoch  11: Loss 0.90173, Train accuracy 0.683, Test accuracy 0.648, Throughput 658.01 images/second,  Time 79.318 (s)
Epoch  12: Loss 0.87864, Train accuracy 0.684, Test accuracy 0.654, Throughput 665.73 images/second,  Time 78.139 (s)
Epoch  13: Loss 0.85572, Train accuracy 0.693, Test accuracy 0.664, Throughput 656.16 images/second,  Time 79.492 (s)
Epoch  14: Loss 0.84468, Train accuracy 0.702, Test accuracy 0.674, Throughput 662.35 images/second,  Time 78.570 (s)
Epoch  15: Loss 0.82761, Train accuracy 0.720, Test accuracy 0.681, Throughput 650.20 images/second,  Time 80.583 (s)
Epoch  16: Loss 0.80679, Train accuracy 0.718, Test accuracy 0.676, Throughput 628.71 images/second,  Time 82.960 (s)
Epoch  17: Loss 0.79770, Train accuracy 0.730, Test accuracy 0.694, Throughput 637.80 images/second,  Time 81.773 (s)
Epoch  18: Loss 0.77807, Train accuracy 0.738, Test accuracy 0.697, Throughput 649.98 images/second,  Time 80.138 (s)
Epoch  19: Loss 0.76274, Train accuracy 0.739, Test accuracy 0.695, Throughput 656.98 images/second,  Time 78.802 (s)
```
