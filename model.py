# Markus Enzweiler - markus.enzweiler@hs-esslingen.de

import os

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

import norm


class CNN(nn.Module):
    """A simple CNN for CIFAR-10 / CIFAR-100."""

    def __init__(self, num_classes=10):
        super().__init__()

        self.num_classes = num_classes

        # mlx=0.0.6 does not have MaxPool2d yet, so we use stride=2 in Conv2d instead
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Poor man's ResNet ...
        # skip connections (learned 1x1 convolutions with stride=2 to match dimensions)
        self.conv_skip2 = nn.Conv2d(32, 32, 1, stride=1, padding=0)
        self.conv_skip4 = nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.conv_skip6 = nn.Conv2d(128, 128, 1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm(32)
        self.bn2 = nn.BatchNorm(32)
        self.bn3 = nn.BatchNorm(64)
        self.bn4 = nn.BatchNorm(64)
        self.bn5 = nn.BatchNorm(128)
        self.bn6 = nn.BatchNorm(128)

        self.fc1 = nn.Linear(4 * 4 * 128, 128)
        self.fc2 = nn.Linear(128, self.num_classes)

        self.drop = nn.Dropout(0.25)

    def __call__(self, x):
        # Poor man's ResNet with residual connections

        # For some reason, residual connections work better in this
        # example with relu() applied before the addition and not after
        # the addition as in the original ResNet paper.

        # Input 32x32x3  | Output 16x16x32
        x = nn.relu(self.bn1(self.conv1(x)))
        x = self.conv_skip2(x) + nn.relu(self.bn2(self.conv2(x)))  # residual connection
        # x = nn.relu (self.conv_skip2(x) + self.conv2(x)) # residual connection
        x = self.pool(x)
        x = self.drop(x)

        # Input 16x16x32 | Output 8x8x64
        x = nn.relu(self.bn3(self.conv3(x)))
        x = self.conv_skip4(x) + nn.relu(self.bn4(self.conv4(x)))  # residual connection
        # x = nn.relu (self.conv_skip4(x) + self.conv4(x)) # residual connection
        x = self.pool(x)
        x = self.drop(x)

        # Input 8x8x64 | Output 4x4x128
        x = nn.relu(self.bn5(self.conv5(x)))
        x = self.conv_skip6(x) + nn.relu(self.bn6(self.conv6(x)))  # residual connection
        # x = nn.relu (self.conv_skip6(x) + self.conv6(x)) # residual connection
        x = self.pool(x)
        x = self.drop(x)

        # MLP classifier on top
        # Flatten:  Input 4x4x128  | Output 2048 (4*4*128)
        x = x.reshape(x.shape[0], -1)  # no flatten() available
        # Input 2048  | Output 128
        x = nn.relu(self.fc1(x))
        # Input 128   | Output num_classes
        x = self.fc2(x)  # no activation, cross_entropy loss applies softmax
        return x

    def num_params(self):
        nparams = sum(x.size for k, x in tree_flatten(self.trainable_parameters()))
        return nparams

    def save(self, fname):
        # Extract the directory path from the file name
        dir_path = os.path.dirname(fname)

        # Check if the directory path is not empty
        if dir_path:
            # Check if the directory exists, and create it if it does not
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)

        # save the model weights
        self.save_weights(fname)

    def load(self, fname):
        self.load_weights(fname)
