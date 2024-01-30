from torch.utils.data import Dataset
import numpy as np
import torch

class AudioDataset(Dataset):
    def __init__(self, amplitudes: np.ndarray, width: int, height: int, alpha: float, device):
        self.device = device
        self.size = amplitudes.shape[0]
        self.width = width
        self.height = height
        self.alpha = alpha

        self.features = []

        self.rowmat = np.tile(np.linspace(-1, 1, self.height), self.width).reshape(self.width, self.height).T
        self.colmat = np.tile(np.linspace(-1, 1, self.width), self.height).reshape(self.height, self.width)

        feature = amplitudes[0, :]

        for t in range(self.size):
            feature = self.alpha*feature + (1-self.alpha)*amplitudes[t, :]
            self.features.append(feature)


    def __len__(self):
        return self.size

    def __getitem__(self, index):
        fmaps = [f*np.ones(self.rowmat.shape) for f in self.features[index]]
        inputs = [self.rowmat, self.colmat, np.sqrt(np.power(self.rowmat, 2)+np.power(self.colmat, 2))]
        inputs.extend(fmaps)

        coordmat = np.stack(inputs).transpose(1, 2, 0)
        coordmat = coordmat.reshape(-1, coordmat.shape[2])

        return torch.from_numpy(coordmat.astype(np.float32)).to(self.device)
    