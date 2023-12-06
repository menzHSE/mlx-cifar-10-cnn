# Markus Enzweiler - markus.enzweiler@hs-esslingen.de

# This is a simple CNN for CIFAR-10 to tryout mlx. 
# It heavily borrows from the mnist example in https://github.com/ml-explore/mlx-examples

import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


def cifar10(save_dir="./data"):

    # Initialize a dictionary to store the CIFAR10 data
    cifar10Data = {}

    # Use torchvision to download the dataset 
    # https://github.com/ml-explore/mlx-data might come in handy in the future
    cifar10TrainSet = torchvision.datasets.CIFAR10(root=save_dir, train=True,  download=True, transform=None)
    cifar10TestSet  = torchvision.datasets.CIFAR10(root=save_dir, train=False, download=True, transform=None)

    # Load the CIFAR10 training and test datasets

    # Convert training data to NumPy arrays
    cifar10Data["training_images"] = np.array([np.array(img).reshape(32, 32, 3) for img, _ in cifar10TrainSet])
    cifar10Data["training_labels"] = np.array([label for _, label in cifar10TrainSet])

    # Convert test data to NumPy arrays
    cifar10Data["test_images"] = np.array([np.array(img).reshape(32, 32, 3) for img, _ in cifar10TestSet])
    cifar10Data["test_labels"] = np.array([label for _, label in cifar10TestSet])

    # Normalize to 0-1
    preproc = lambda x: x.astype(np.float32) / 255.0
    cifar10Data["training_images"] = preproc(cifar10Data["training_images"])
    cifar10Data["test_images"]     = preproc(cifar10Data["test_images"])

    return (
        cifar10Data["training_images"],
        cifar10Data["training_labels"].astype(np.uint32),
        cifar10Data["test_images"],
        cifar10Data["test_labels"].astype(np.uint32),
    )


if __name__ == "__main__":
    train_x, train_y, test_x, test_y = cifar10()
    assert train_x.shape == (50000, 32, 32, 3), "Wrong training set size"
    assert train_y.shape == (50000,), "Wrong training set size"
    assert test_x.shape  == (10000, 32, 32, 3), "Wrong test set size"
    assert test_y.shape  == (10000,), "Wrong test set size"

    # Save an image as a sanity check
    image_data = (train_x[123] * 255).astype(np.uint8)
    img = Image.fromarray(image_data)
    img.save("/tmp/train123.png")

    print("Dataset prepared successfully!")