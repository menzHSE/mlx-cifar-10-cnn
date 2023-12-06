# mlx-cifar-10-cnn
CIFAR-10 CNN implementation in Apple mlx. Used to try-out the first public version of https://github.com/ml-explore/mlx
It heavily borrows from the mnist example in https://github.com/ml-explore/mlx-examples

# Requirements
* Machine with Apple silicon (M1/M2/M3)
* Install https://github.com/ml-explore/mlx
* torchvision

# Run
`python cnn.py --gpu`

```
(mlx-m1-2023-12) $ python cnn.py  --gpu
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz
100.0%
Extracting ./data/cifar-10-python.tar.gz to ./data
Starting training ...
Epoch 0: Loss 1.69997, Train accuracy 0.530, Test accuracy 0.496, Time 27.981 (s)
Epoch 1: Loss 1.31770, Train accuracy 0.597, Test accuracy 0.543, Time 28.509 (s)
Epoch 2: Loss 1.16119, Train accuracy 0.647, Test accuracy 0.578, Time 28.207 (s)
Epoch 3: Loss 1.02115, Train accuracy 0.708, Test accuracy 0.598, Time 28.457 (s)
Epoch 4: Loss 0.89512, Train accuracy 0.759, Test accuracy 0.612, Time 29.385 (s)
Epoch 5: Loss 0.78448, Train accuracy 0.783, Test accuracy 0.610, Time 29.438 (s)
Epoch 6: Loss 0.68474, Train accuracy 0.810, Test accuracy 0.611, Time 29.130 (s)
Epoch 7: Loss 0.59645, Train accuracy 0.818, Test accuracy 0.591, Time 28.030 (s)
Epoch 8: Loss 0.51680, Train accuracy 0.884, Test accuracy 0.616, Time 28.503 (s)
Epoch 9: Loss 0.44857, Train accuracy 0.887, Test accuracy 0.598, Time 27.958 (s)
Test accuracy 0.598
```
