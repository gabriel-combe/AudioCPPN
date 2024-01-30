import cv2
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from typing import List
from torch.utils.data import DataLoader

from AudioProcessing import extractAudio, stft, preprocessAmplitudes
from utils import createFolder, createVideo, getDevice
from DatasetGenerator import AudioDataset
from CPPNModel import CPPN, init_weights

AUDIOPATH = "audios/Dream_Fiction_-_Rhodz.mp3"
VIDEOPATH = "videos/Dream_Fiction_-_Rhodz.mp4"

FPS = 30
WIDTH = 512
HEIGHT = 512

GAIN = 0.5
ALPHA = 0.8

NLAYERS = 8
HSIZE = 32
OUTPUTSIZE = 3

if __name__ == '__main__':
    # Get the device on which to run the model
    device = getDevice()

    # Get the audio data from the audio file
    sound, fs = extractAudio(AUDIOPATH)

    # Extract the amplitudes (frequencies) of the audio
    amplitudes = stft(sound, fs, FPS, 1024)

    # Cleanup the amplitudes
    processedAmplitudes = preprocessAmplitudes(amplitudes, GAIN)

    model = CPPN(device, processedAmplitudes.shape[1], NLAYERS, HSIZE, OUTPUTSIZE)
    model.apply(init_weights)
    model.to(device)

    dataset = AudioDataset(processedAmplitudes, WIDTH, HEIGHT, ALPHA, device)
    dataloader = DataLoader(dataset, batch_size=64)

    createFolder()

    nb = 0

    with torch.no_grad():
        for data in tqdm(dataloader, desc='Generate Frames'):
            result = model(data).cpu()

            for res in result:
                frame = res.numpy().reshape(HEIGHT, WIDTH, -1).astype(np.uint8)
                cv2.imshow('...', frame)
                cv2.imwrite(f'frames/{nb:06d}.png', frame)
                cv2.waitKey(1)
                nb += 1
    
    cv2.destroyAllWindows()

    createVideo(AUDIOPATH, VIDEOPATH, FPS, WIDTH, HEIGHT)

