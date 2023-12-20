# Markus Enzweiler - markus.enzweiler@hs-esslingen.de

# This is a simple CNN for CIFAR-10 to tryout mlx. 
# It heavily borrows from the cifar example in https://github.com/ml-explore/mlx-examples

import argparse
import time
import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import dataset
import model


def loss_fn(model, X, y):
    predictions = model(X)
    loss = mx.mean(nn.losses.cross_entropy(predictions, y))
    acc  = mx.mean(mx.argmax(predictions, axis=1) == y)
    return loss, acc

def eval_fn(model, X, y):   
    return mx.mean(mx.argmax(model(X), axis=1) == y)

def test_epoch(model, data_iter):
    accs = []
    for batch_counter, batch in enumerate(data_iter):
        X = mx.array(batch["image"])
        y = mx.array(batch["label"])
        acc = eval_fn(model, X, y)
        acc_value = acc.item()
        accs.append(acc_value)
    mean_acc = mx.mean(mx.array(accs))
    return mean_acc

def save_model(cnn, name, epoch):
    fname = f"{name}_{epoch:03d}.npz"
    cnn.save(fname)

def train_epoch(model, tr_iter, loss_and_grad_fn, optimizer, epoch):
   
    running_loss = 0.0
    running_acc = 0.0
    throughput_list = []

    # iterate over training batches
    for batch_count, batch in enumerate(tr_iter):
        X = mx.array(batch["image"])
        y = mx.array(batch["label"])

        throughput_tic = time.perf_counter()

        # forward pass + backward pass + update
        (loss, acc), grads  = loss_and_grad_fn(model, X, y)
        optimizer.update(model, grads)
        # Evaluate updated model parameters
        mx.eval(model.parameters(), optimizer.state)

        throughput_toc = time.perf_counter()
        throughput = X.shape[0] / (throughput_toc - throughput_tic)
        throughput_list.append(throughput)
        running_loss += loss
        running_acc += acc

        if batch_count > 0 and (batch_count % 10 == 0):
            print(f"Epoch {epoch:4d}: Loss {running_loss.item() / batch_count:.5f}, "
                  f"Batch {batch_count:5d}, "
                  f"Train accuracy {running_acc.item() / batch_count:.5f}, "
                  f"Throughput {throughput:.2f} images/second", end="\r")

        batch_count = batch_count + 1

        #### end of loop over training batches ####

    return running_loss, throughput_list, batch_count

    

def train(batch_size, num_epochs, learning_rate):
       
    # Load the training and test data
    tr_iter, test_iter = dataset.cifar10(batch_size)

    # Load the model
    cnn = model.CNN()
    # Allocate memory and initialize parameters
    mx.eval(cnn.parameters())
    #print(cnn.modules())

    loss_and_grad_fn = nn.value_and_grad(cnn, loss_fn)
    optimizer = optim.AdamW(learning_rate=learning_rate)
    #optimizer = optim.Adagrad(learning_rate=learning_rate)

    print("Starting training ...")

    for e in range(num_epochs):
        # reset iterators and stats at the beginning of each epoch
        tr_iter.reset()
        test_iter.reset()

        # train one epoch
        tic = time.perf_counter()
        running_loss, throughput_list, batch_count = train_epoch(cnn, tr_iter, loss_and_grad_fn, optimizer, e)
        toc = time.perf_counter()
               
        # reset iterators before testing
        tr_iter.reset()
        test_iter.reset()
        train_accuracy = test_epoch(cnn, tr_iter)
        test_accuracy  = test_epoch(cnn, test_iter)       
        samples_per_sec = mx.mean(mx.array(throughput_list))

        # print stats
        print(
            f"Epoch {e:4d}: Loss {running_loss.item() / batch_count:.5f}, Train accuracy {train_accuracy.item():.3f}, "
            f"Test accuracy {test_accuracy.item():.3f}, "           
            f"Throughput {samples_per_sec.item():.2f} images/second, ",
            f"Time {toc - tic:.3f} (s)"
        )     

        # save model
        save_model(cnn, "model", e)       


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a simple CNN on CIFAR-10 with mlx.")

    parser.add_argument("--gpu", action="store_true", help="Use Metal GPU acceleration")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--batchsize", type=int, default=64, help="Batch size for training")
    parser.add_argument("--nepoch", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
     
    args = parser.parse_args()

    if not args.gpu:
        mx.set_default_device(mx.cpu)

    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    print("Options: ")
    print(f"  GPU: {args.gpu}")
    print(f"  Seed: {args.seed}")
    print(f"  Batch size: {args.batchsize}")
    print(f"  Number of epochs: {args.nepoch}")
    print(f"  Learning rate: {args.lr}")

    train(args.batchsize, args.nepoch, args.lr)
