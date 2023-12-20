# Markus Enzweiler - markus.enzweiler@hs-esslingen.de

import argparse
import time
import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mutils
import dataset
import model

def eval_fn(cnn, X, y):   
    return mx.mean(mx.argmax(cnn(X), axis=1) == y)

def test(model_fname):
       
    # Load the training and test data
    _, test_iter = dataset.cifar10(batch_size=32)

    # Load the model
    cnn = model.CNN()
    cnn.load(model_fname)
    print(f"Loaded model from {model_fname}")

    # Evaluate the model on the test set
    accs = []
    print("Starting testing ...")
    for batch_counter, batch in enumerate(test_iter):
        if batch_counter % 100 == 0:
            print(f".", end="", flush=True)
        X = mx.array(batch["image"])
        y = mx.array(batch["label"])
        acc = eval_fn(cnn, X, y)       
        acc_value = acc.item()      
        accs.append(acc_value)
    print("")
    mean_acc = mx.mean(mx.array(accs))
    print(f"Test accuracy: {mean_acc.item()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test a simple CNN on CIFAR-10 with mlx.")

    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of Metal GPU acceleration")     
    parser.add_argument('--model', type=str, required=True, help='Model filename *.npz')

    args = parser.parse_args()

    if args.cpu:
        mx.set_default_device(mx.cpu)

    test(args.model)
