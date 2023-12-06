# mlx-cifar-10-cnn
Small CIFAR-10 CNN implementation in Apple mlx. Used to try-out the first public version of https://github.com/ml-explore/mlx
It heavily borrows from the mnist example in https://github.com/ml-explore/mlx-examples

# Requirements
* Machine with Apple silicon (M1/M2/M3)
* mlx (tested with version 0.0.3). Install using instructions on https://github.com/ml-explore/mlx
* torchvision (for handling CIFAR-10 data)

# Limitations
* mlx does not yet include pooling layers, see https://github.com/ml-explore/mlx/issues/25

# Run
`python cnn.py --gpu`

```
(mlx-m1-2023-12) $ python cnn.py  --gpu
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz
100.0%
Extracting ./data/cifar-10-python.tar.gz to ./data
Starting training ...
Epoch 0: Loss 1.53577, Train accuracy 0.530, Test accuracy 0.523, Time 27.800 (s)
Epoch 1: Loss 1.27640, Train accuracy 0.582, Test accuracy 0.565, Time 28.480 (s)
Epoch 2: Loss 1.16963, Train accuracy 0.607, Test accuracy 0.583, Time 20.685 (s)
Epoch 3: Loss 1.09031, Train accuracy 0.644, Test accuracy 0.612, Time 21.257 (s)
Epoch 4: Loss 1.02532, Train accuracy 0.668, Test accuracy 0.630, Time 20.396 (s)
Epoch 5: Loss 0.96800, Train accuracy 0.677, Test accuracy 0.625, Time 20.876 (s)
Epoch 6: Loss 0.92005, Train accuracy 0.697, Test accuracy 0.633, Time 20.771 (s)
Epoch 7: Loss 0.87206, Train accuracy 0.697, Test accuracy 0.631, Time 20.479 (s)
Epoch 8: Loss 0.83043, Train accuracy 0.731, Test accuracy 0.650, Time 21.005 (s)
Epoch 9: Loss 0.79088, Train accuracy 0.750, Test accuracy 0.660, Time 20.711 (s)
Epoch 10: Loss 0.75901, Train accuracy 0.756, Test accuracy 0.652, Time 20.860 (s)
Epoch 11: Loss 0.72243, Train accuracy 0.773, Test accuracy 0.660, Time 20.788 (s)
Epoch 12: Loss 0.68936, Train accuracy 0.789, Test accuracy 0.665, Time 20.514 (s)
Epoch 13: Loss 0.65548, Train accuracy 0.794, Test accuracy 0.663, Time 20.826 (s)
Epoch 14: Loss 0.62693, Train accuracy 0.814, Test accuracy 0.663, Time 21.430 (s)
Test accuracy 0.663
```
