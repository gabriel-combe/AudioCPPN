from torch.utils.data import Dataset
import numpy as np
import torch

class AudioDataset(Dataset):
    def __init__(self, amplitudes: np.ndarray, heightMapArray: np.ndarray, width: int, height: int, alpha: float, device):
        self.device = device
        self.size = amplitudes.shape[0]
        self.width = width
        self.height = height
        self.alpha = alpha
        self.heightmap = heightMapArray

        self.heightmapCount = self.heightmap.shape[0]
        self.heightmap = self.heightmap.transpose(0, 2, 3, 1)
        self.heightmap = self.heightmap.reshape(self.heightmapCount, -1, 3)

        self.features = []

        self.coordmat = np.zeros((3+amplitudes.shape[1], self.height, self.width))
        self.coordmat = self.coordmat.transpose(1, 2, 0)
        self.coordmat = self.coordmat.reshape(-1, self.coordmat.shape[2])

        feature = amplitudes[0, :]

        for t in range(self.size):
            feature = self.alpha*feature + (1-self.alpha)*amplitudes[t, :]
            self.features.append(feature)


    def __len__(self):
        return self.size

    def __getitem__(self, index):

        self.coordmat[:, :3] = self.heightmap[min(index, self.heightmapCount-1)]
        self.coordmat[:, 3:] = self.features[index]

        return torch.from_numpy(self.coordmat.astype(np.float32)).to(self.device)
    