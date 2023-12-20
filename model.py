# Markus Enzweiler - markus.enzweiler@hs-esslingen.de

import mlx.nn as nn

class CNN(nn.Module):
    """A simple CNN for CIFAR-10. """

    def __init__(self):
        super().__init__()
      
        self.conv1 = nn.Conv2d(3,        32,  3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32,       64,  3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64,      128,  3, stride=2, padding=1)

        self.fc1   = nn.Linear(4*4*128,  128                        )
        self.fc2   = nn.Linear(128,      10                         )

        self.drop  = nn.Dropout(0.25)

        self.ln1   = nn.LayerNorm(32)
        self.ln2   = nn.LayerNorm(64)
        self.ln3   = nn.LayerNorm(128)


    def __call__(self, x):
      
        # Input 32x32x3  | Output 16x16x32
        x = nn.relu   (self.ln1(self.conv1(x)))        
        x = self.drop (x)
       
        # Input 16x16x32 | Output 8x8x64
        x = nn.relu   (self.ln2(self.conv2(x)))
        x = self.drop (x)

        # Input 8x8x64 | Output 4x4x128
        x = nn.relu   (self.ln3(self.conv3(x)))     
        x = self.drop (x)        

        # MLP classifier on top
        # Flatten:  Input 4x4x128  | Output 2048 (4*4*128)
        x = x.reshape (x.shape[0], -1) # no flatten() available
        # Input 2048  | Output 128
        x = nn.relu   (self.fc1(x))     
        # Input 128   | Output 10    
        x = self.fc2  (x) # no activation, cross_entropy loss applies softmax 
        return x

    def save(self, fname):
       self.save_weights(fname)

    def load(self, fname):
       self.load_weights(fname)