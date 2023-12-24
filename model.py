# Markus Enzweiler - markus.enzweiler@hs-esslingen.de

import mlx.nn as nn
from mlx.utils import tree_flatten

class CNN(nn.Module):
    """A simple CNN for CIFAR-10 / CIFAR-100. """

    def __init__(self, num_classes=10):
        super().__init__()

        self.num_classes = num_classes

        # mlx=0.0.6 does not have MaxPool2d yet, so we use stride=2 in Conv2d instead        
        self.conv1 = nn.Conv2d   (3,        32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d   (32,       32, 3, stride=2, padding=1)                    
        self.conv3 = nn.Conv2d   (32,       64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d   (64,       64, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d   (64,      128, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d   (128,     128, 3, stride=2, padding=1)

        # mlx=0.0.6 does not have BatchNorm yet, so we use LayerNorm instead
        self.ln1  = nn.LayerNorm( 32)
        self.ln2  = nn.LayerNorm( 32)
        self.ln3  = nn.LayerNorm( 64)
        self.ln4  = nn.LayerNorm( 64)
        self.ln5  = nn.LayerNorm(128)
        self.ln6  = nn.LayerNorm(128)

        self.fc1   = nn.Linear(4*4*128,  128             )
        self.fc2   = nn.Linear(128,      self.num_classes)

        self.drop  = nn.Dropout(0.25)



    def __call__(self, x):
      
        # Input 32x32x3  | Output 16x16x32
        x = nn.relu   (self.ln1(self.conv1(x)))        
        x = nn.relu   (self.ln2(self.conv2(x)))        
        x = self.drop (x)
       
        # Input 16x16x32 | Output 8x8x64
        x = nn.relu   (self.ln3(self.conv3(x)))        
        x = nn.relu   (self.ln4(self.conv4(x)))        
        x = self.drop (x)

        # Input 8x8x64 | Output 4x4x128
        x = nn.relu   (self.ln5(self.conv5(x)))        
        x = nn.relu   (self.ln6(self.conv6(x)))        
        x = self.drop (x)        

        # MLP classifier on top
        # Flatten:  Input 4x4x128  | Output 2048 (4*4*128)
        x = x.reshape (x.shape[0], -1) # no flatten() available
        # Input 2048  | Output 128
        x = nn.relu   (self.fc1(x))     
        # Input 128   | Output num_classes    
        x = self.fc2  (x) # no activation, cross_entropy loss applies softmax 
        return x
    
    def num_params(self):
        nparams = sum(x.size for k, x in tree_flatten(self.trainable_parameters()))
        return nparams

    def save(self, fname):
        self.save_weights(fname)

    def load(self, fname):
        self.load_weights(fname)