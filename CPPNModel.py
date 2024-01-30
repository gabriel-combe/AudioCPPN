import numpy as np
import torch.nn as nn

class CPPN(nn.Module):
    def __init__(self, device, amplitudesSize: int =8, nlayers: int =8, hsize: int =16, outputSize: int =3):
        super(CPPN, self).__init__()
        self.device = device

        self.linearLayers = [nn.Linear(3 + amplitudesSize, hsize, bias=False, device=self.device)]

        for layer in range(1,nlayers-1):
            self.linearLayers.append(nn.Linear(hsize, hsize, bias=False, device=self.device))
        
        self.linearLayers.append(nn.Linear(hsize, outputSize, bias=False, device=self.device))

        self.activation = nn.Tanh()
    
    def forward(self, x):
        print(x.shape)
        for layer in self.linearLayers:
            x = self.activation(layer(x))
        
        return (1.0 + x)/2.0

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight)