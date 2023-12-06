# mlx-cifar-10-cnn
CIFAR-10 CNN implementation in Apple mlx. Used to try-out the first public version of https://github.com/ml-explore/mlx
It heavily borrows from the mnist example in https://github.com/ml-explore/mlx-examples

# Requirements
* Machine with Apple silicon (M1/M2/M3)
* mlx (tested with version 0.0.3). Install using instructions on https://github.com/ml-explore/mlx
* torchvision (for handling CIFAR-10)

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
Epoch 0: Loss 1.53476, Train accuracy 0.528, Test accuracy 0.523, Time 25.004 (s)
Epoch 1: Loss 1.27585, Train accuracy 0.585, Test accuracy 0.568, Time 23.399 (s)
Epoch 2: Loss 1.16578, Train accuracy 0.611, Test accuracy 0.591, Time 27.832 (s)
Epoch 3: Loss 1.08704, Train accuracy 0.644, Test accuracy 0.611, Time 28.561 (s)
Epoch 4: Loss 1.02331, Train accuracy 0.668, Test accuracy 0.627, Time 28.774 (s)
Epoch 5: Loss 0.96585, Train accuracy 0.679, Test accuracy 0.624, Time 29.023 (s)
Epoch 6: Loss 0.91720, Train accuracy 0.696, Test accuracy 0.631, Time 28.765 (s)
Epoch 7: Loss 0.86825, Train accuracy 0.703, Test accuracy 0.637, Time 27.920 (s)
Epoch 8: Loss 0.82499, Train accuracy 0.734, Test accuracy 0.649, Time 28.059 (s)
Epoch 9: Loss 0.78654, Train accuracy 0.752, Test accuracy 0.655, Time 28.645 (s)
Test accuracy 0.655
```
