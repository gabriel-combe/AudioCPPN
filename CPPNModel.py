import numpy as np
import torch.nn as nn

class CPPN(nn.Module):
    def __init__(self, device, amplitudesSize: int =8, nlayers: int =8, hsize: int =16, outputSize: int =3):
        super(CPPN, self).__init__()
        self.device = device

        self.linearLayers = nn.ModuleList([nn.Linear(3 + amplitudesSize, hsize, bias=False)])
        self.linearLayers.extend([nn.Linear(hsize, hsize, bias=False) for _ in range(1, nlayers-1)])
        self.linearLayers.append(nn.Linear(hsize, outputSize, bias=False))

        self.activation = nn.Tanh()
    
    def forward(self, x):
        for layer in self.linearLayers:
            x = self.activation(layer(x))
        
        return 255.0 * (1.0 + x)/2.0

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight)