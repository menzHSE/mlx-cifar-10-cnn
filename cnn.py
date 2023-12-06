# Markus Enzweiler - markus.enzweiler@hs-esslingen.de

# This is a simple CNN for CIFAR-10 to tryout mlx. 
# It heavily borrows from the mnist example in https://github.com/ml-explore/mlx-examples

import argparse
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import dataset

class CNN(nn.Module):
    """A simple CNN for CIFAR-10. """

    def __init__(self):
        super().__init__()

        # No MaxPool2D in mlx (yet), so we use stride=2 in Conv2D instead (not the same, but ...)
      
        # Input 32x32x3     Output 16x16x32
        self.conv1 = nn.Conv2d(3,        32, 3,  stride=2, padding=1)
        # Input 16x16x32    Output 8x8x64
        self.conv2 = nn.Conv2d(32,       64, 3,  stride=2, padding=1)       
        self.fc1 =   nn.Linear(64*8*8,   64                         )
        self.fc2 =   nn.Linear(64,       10                         )
        

    def __call__(self, x):
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))      
        x = x.reshape(x.shape[0], -1) # no flatten() available
        x = nn.relu(self.fc1(x))        
        x = self.fc2(x) # no activation, cross_entropy loss does that for us
        return x

def loss_fn(model, X, y):
    return mx.mean(nn.losses.cross_entropy(model(X), y))


def eval_fn(model, X, y):
    return mx.mean(mx.argmax(model(X), axis=1) == y)


def batch_iterate(batch_size, X, y):
    perm = mx.array(np.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s : s + batch_size]
        yield X[ids], y[ids]


def main():
    seed = 0     
    batch_size = 16
    num_epochs = 10
    learning_rate = 3e-4
    np.random.seed(seed)

    # Load the data
    train_images, train_labels, test_images, test_labels = map(mx.array, dataset.cifar10())

    # Load the model
    model = CNN()
    mx.eval(model.parameters())

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.Adam(learning_rate=learning_rate)

    print("Starting training ...")

    for e in range(num_epochs):
        tic = time.perf_counter()
        running_loss = 0.0
        batch_count = 0
        for X, y in batch_iterate(batch_size, train_images, train_labels):
            loss, grads = loss_and_grad_fn(model, X, y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            running_loss += loss
            batch_count = batch_count + 1
        train_accuracy = eval_fn(model, train_images, train_labels)
        test_accuracy  = eval_fn(model, test_images,  test_labels)
        toc = time.perf_counter()
        print(
            f"Epoch {e}: Loss {running_loss.item() / batch_count:.5f}, Train accuracy {train_accuracy.item():.3f}, Test accuracy {test_accuracy.item():.3f},"
            f" Time {toc - tic:.3f} (s)"
        )       
        batch_count = 0
        running_loss = 0.0

    # final eval
    test_accuracy  = eval_fn(model, test_images,  test_labels)
    print(f"Test accuracy {test_accuracy.item():.3f}")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a simple CNN on CIFAR-10 with MLX.")
    parser.add_argument("--gpu", action="store_true", help="Use the Metal back-end.")
    args = parser.parse_args()
    if not args.gpu:
        mx.set_default_device(mx.cpu)
    main()