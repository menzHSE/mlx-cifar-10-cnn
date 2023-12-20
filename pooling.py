from typing import Tuple, Union
import mlx.core as mx
import mlx.nn as nn
    
class AvgPool2D(nn.Module):
    def __init__(
        self,
        stride: int        
    ):
        super().__init__()      
        self.stride = stride
       

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, C = x.shape
        x = x.reshape(B, W//self.stride, self.stride, H//self.stride, self.stride, C).mean((2, 4))
        return x
    

if __name__ == "__main__":
    s = 4
    t = mx.arange(4*(s**2)).reshape(2, s, s, 2)
    mp = AvgPool2D(2)
    print(t[..., 0])
    pooled = mp(t)
    print(pooled[..., 0])
    print(pooled.shape)
