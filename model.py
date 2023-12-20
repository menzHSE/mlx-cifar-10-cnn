import mlx.nn as nn
import pooling as pool

class CNN(nn.Module):
    """A simple CNN for CIFAR-10. """

    def __init__(self):
        super().__init__()

        # No MaxPool2D in mlx (yet), so we use stride=2 in Conv2D instead (not the same, but ...)
      
        # Input 32x32x3  | Output 16x16x32
        # Input 16x16x32 | Output 8x8x64
        self.conv1 = nn.Conv2d(3,       32,  3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32,      32,  3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32,      64,  3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64,      64,  3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64,      128, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128,     128, 3, stride=1, padding=1)

        self.fc1   = nn.Linear(4*4*128, 128                        )
        self.fc2   = nn.Linear(128,     10                         )

        self.pool  = pool.AvgPool2D(2)
        self.drop  = nn.Dropout(0.33)

    def __call__(self, x):

        # First block
        # Input 32x32x3  | Output 32x32x32
        x = nn.relu   (self.conv1(x))
        # Input 32x32x32 | Output 32x32x32
        x = nn.relu   (self.conv2(x))
        # Input 32x32x32 | Output 16x16x32
        x = self.drop (self.pool(x))

        # Second block
        # Input 16x16x32 | Output 16x16x64
        x = nn.relu   (self.conv3(x))
        # Input 16x16x64 | Output 16x16x64
        x = nn.relu   (self.conv4(x))
        # Input 16x16x64 | Output 8x8x64
        x = self.drop (self.pool(x))

        # Third block
        # Input 8x8x64   | Output 8x8x128
        x = nn.relu   (self.conv5(x))
        # Input 8x8x128  | Output 8x8x128
        x = nn.relu   (self.conv6(x))
        # Input 8x8x128  | Output 4x4x128
        x = self.drop (self.pool(x))

        # MLP classifier on top
        # Flatten:  Input 4x4x128  | Output 2048 (4*4*128)
        x = x.reshape (x.shape[0], -1) # no flatten() available
        # Input 2048  | Output 128
        x = nn.relu   (self.fc1(x))    
        # Input 128   | Output 10    
        x = self.fc2  (x) # no activation, cross_entropy loss applies softmax 
        return x

    def save(self, filename):      
      self.save_weights(filename)