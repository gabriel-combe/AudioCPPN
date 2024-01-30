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
HSIZE = 16
OUTPUTSIZE = 3

def buildCPPN(amplitudesSize: int, rng) -> List[np.ndarray]:
    # Create CPPN layers
    cppn = [rng.standard_normal(size=(3 + amplitudesSize, HSIZE))]

    for i in range(1,NLAYERS-1):
        cppn.append(rng.standard_normal(size=(HSIZE, HSIZE)))

    cppn.append(rng.standard_normal(size=(HSIZE, OUTPUTSIZE)))

    return cppn

def genImage(features: np.ndarray, cppn: List[np.ndarray], rowmat: np.ndarray, colmat: np.ndarray) -> np.ndarray:

    fmaps = [f*np.ones(rowmat.shape) for f in features]

    inputs = [rowmat, colmat, np.sqrt(np.power(rowmat, 2)+np.power(colmat, 2))]
    inputs.extend(fmaps)

    coordmat = np.stack(inputs).transpose(1, 2, 0)
    coordmat = coordmat.reshape(-1, coordmat.shape[2])

    result = coordmat.copy()

    for layer in cppn:
        result = np.tanh(np.matmul(result, layer))
    
    result = (1.0 + result)/2.0

    return result

if __name__ == '__main__':
    # Get the device on which to run the model
    device = getDevice()

    # Get the audio data from the audio file
    sound, fs = extractAudio(AUDIOPATH)

    # Extract the amplitudes (frequencies) of the audio
    amplitudes = stft(sound, fs, FPS, 2048)

    # Cleanup the amplitudes
    processedAmplitudes = preprocessAmplitudes(amplitudes, GAIN)

    rowmat = np.tile(np.linspace(-1, 1, HEIGHT), WIDTH).reshape(WIDTH, HEIGHT).T
    colmat = np.tile(np.linspace(-1, 1, WIDTH), HEIGHT).reshape(HEIGHT, WIDTH)

    rng = np.random.default_rng(seed=None)

    cppn = buildCPPN(processedAmplitudes.shape[1], rng)
    model = CPPN(device, processedAmplitudes.shape[1], NLAYERS, HSIZE, OUTPUTSIZE)
    model.apply(init_weights)
    model.to(device)

    dataset = AudioDataset(processedAmplitudes, WIDTH, HEIGHT, ALPHA, device)
    dataloader = DataLoader(dataset, batch_size=1)

    # createFolder()

    features = processedAmplitudes[0, :]
    features = ALPHA*features + (1-ALPHA)*processedAmplitudes[0, :]
    frame = genImage(features, cppn, rowmat, colmat)
    frame = (255.0*frame.reshape(HEIGHT, WIDTH, -1)).astype(np.uint8)

    cv2.imshow('...', frame)
    cv2.waitKey(2000)

    with torch.no_grad():
        result = model(dataset[0])
    frame = (255.0*result.cpu().numpy().reshape(HEIGHT, WIDTH, -1)).astype(np.uint8)
    cv2.imshow('...', frame)
    cv2.waitKey(2000)

    # nb = 0

    # with torch.no_grad():
    #     for data in tqdm(dataloader, desc='Generate Frames'):
    #         result = model(data).cpu()

    #         for res in result:
    #             frame = (255.0*res.numpy().reshape(HEIGHT, WIDTH, -1)).astype(np.uint8)
    #             cv2.imwrite(f'frames/{nb:06d}.png', frame)
    #             cv2.waitKey(1)
    #             nb += 1


    nbFrames = processedAmplitudes.shape[0]
    features = processedAmplitudes[0, :]




    # for t in tqdm(range(nbFrames)):
    #     # Exponential smoothing
    #     features = ALPHA*features + (1-ALPHA)*processedAmplitudes[t, :]

    #     # Generate a frame with the features
    #     frame = genImage(features, cppn, rowmat, colmat)
    #     frame = (255.0*frame.reshape(HEIGHT, WIDTH, -1)).astype(np.uint8)

    #     # Show frame with opencv
    #     cv2.imshow('...', frame)

    #     # Save frame
    #     cv2.imwrite(f'frames/{t:06d}.png', frame)

    #     cv2.waitKey(1)
    
    cv2.destroyAllWindows()

    # createVideo()

