from torch.utils.data import Dataset
import numpy as np
import torch

class AudioDataset(Dataset):
    def __init__(self, amplitudes: np.ndarray, heightmapTuple: np.ndarray, width: int, height: int, alpha: float, device):
        self.device = device
        self.size = amplitudes.shape[0]
        self.width = width
        self.height = height
        self.alpha = alpha

        self.yy = heightmapTuple[0]
        self.xx = heightmapTuple[1]
        self.zz = heightmapTuple[2]

        self.yyCount = self.yy.shape[0]
        self.xxCount = self.xx.shape[0]
        self.zzCount = self.zz.shape[0]

        self.yy = self.yy.reshape(self.yyCount, -1)
        self.xx = self.xx.reshape(self.xxCount, -1)
        self.zz = self.zz.reshape(self.zzCount, -1)

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

        self.coordmat[:, 0] = self.yy[min(index, self.yyCount-1)]
        self.coordmat[:, 1] = self.xx[min(index, self.xxCount-1)]
        self.coordmat[:, 2] = self.zz[min(index, self.zzCount-1)]
        self.coordmat[:, 3:] = self.features[index]

        return torch.from_numpy(self.coordmat.astype(np.float32)).to(self.device)
    