# Markus Enzweiler - markus.enzweiler@hs-esslingen.de
# This is a based on the cifar example in mlx-example

import math
import os

import mlx.core as mx
import numpy as np
from mlx.data.datasets import load_cifar10, load_cifar100
from PIL import Image


def cifar(batch_size, cifar_version="CIFAR-10", root=None):
    if cifar_version == "CIFAR-10":
        return cifar_loader(batch_size, load_cifar10, root=root)
    elif cifar_version == "CIFAR-100":
        return cifar_loader(batch_size, load_cifar100, root=root)
    else:
        raise ValueError(f"Unknown CIFAR version: {cifar_version}")


def cifar_loader(batch_size, load_fn, root=None):
    # load train and test sets using mlx-data
    tr = load_fn(root=root, train=True)
    test = load_fn(root=root, train=False)

    # normalize to [0,1]
    def normalize(x):
        return x.astype("float32") / 255.0

    # iterator over training set
    tr_iter = (
        tr.shuffle()
        .to_stream()
        .image_random_h_flip("image", prob=0.5)
        .key_transform("image", normalize)
        .batch(batch_size)
    )

    # iterator over training set
    test_iter = test.to_stream().key_transform("image", normalize).batch(batch_size)
    return tr_iter, test_iter


if __name__ == "__main__":
    batch_size = 32
    tr_iter, test_iter = cifar(batch_size=batch_size)

    batch_tr_iter = next(tr_iter)
    assert batch_tr_iter["image"].shape == (
        batch_size,
        32,
        32,
        3,
    ), "Wrong training set size"
    assert batch_tr_iter["label"].shape == (batch_size,), "Wrong training set size"

    batch_test_iter = next(test_iter)
    assert batch_test_iter["image"].shape == (
        batch_size,
        32,
        32,
        3,
    ), "Wrong training set size"
    assert batch_test_iter["label"].shape == (batch_size,), "Wrong training set size"

    # Save an image as a sanity check

    # Get the image data and normalize it
    img_data = batch_tr_iter["image"][0] * 255
    img_data = img_data.astype(np.uint8)

    # Save the image using Pillow
    img = Image.fromarray(img_data)
    img.save("/tmp/trainTmp.png")

    # Reset the iterators, if necessary
    tr_iter.reset()
    test_iter.reset()

    print("Dataset prepared successfully!")
