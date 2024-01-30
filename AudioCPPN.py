import os
import cv2
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from typing import List
from torch.utils.data import DataLoader

from DatasetGenerator import AudioDataset
from CPPNModel import CPPN

AUDIOPATH = "audios/Dream_Fiction_-_Rhodz.mp3"
VIDEOPATH = "videos/Dream_Fiction_-_Rhodz.mp4"
FS = 44100

FPS = 30
WIDTH = 512
HEIGHT = 512

GAIN = 0.5
ALPHA = 0.8

NLAYERS = 8
HSIZE = 16
OUTPUTSIZE = 3

def extractAudio(audiopath: str) -> np.ndarray:
    sound, fs = torchaudio.load(AUDIOPATH)
    print(sound.shape)
    sound = sound.numpy()[0, :]

    if fs != 44100:
        print(f'fs = {fs} [kHz]')
        print('sample rate should be 44.1 [kHZ] -> aborting ...')
        exit()
    
    return sound

def condenseSpectrum(ampspectrum: np.ndarray) -> np.ndarray:
    # Create 8 bands
    bands = np.zeros(8, dtype=np.float32)

    # The bands are based of https://www.teachmeaudio.com/mixing/techniques/audio-spectrum/
    bands[0] = np.sum(ampspectrum[0:3])     # Sub-bass (0 - 60 Hz)
    bands[1] = np.sum(ampspectrum[3:12])    # Bass (60 - 250 Hz)
    bands[2] = np.sum(ampspectrum[12:23])   # Low midrange (250 - 500 Hz)
    bands[3] = np.sum(ampspectrum[23:93])   # Midrange (500 - 2000 Hz)
    bands[4] = np.sum(ampspectrum[93:186])  # Upper midrange (2000 - 4000 Hz)
    bands[5] = np.sum(ampspectrum[186:278]) # Presence (4000 - 6000 Hz)
    bands[6] = np.sum(ampspectrum[278:928]) # Brilliance (6000 - 20000 Hz)
    bands[7] = np.sum(ampspectrum[928:])    # Over audible (20000 - 22050 Hz)

    return bands

def stft(sound: np.ndarray, fps: int, wsize: int) -> np.ndarray:
    # Length of the audio (points)
    nSamples = len(sound)

    stride = FS//fps

    # Amplitudes of each bands for each segments
    amplitudes = []

    for startIndex in range(0,nSamples,stride):
        # Get the end index
        endIndex = startIndex + wsize
        endIndex = endIndex if endIndex <= nSamples else nSamples
        
        # Retrieve a wsize chunk of audio corresponding to one frame
        chunk = sound[startIndex:endIndex]

        # Pad the chunk if its length is less than the wsize
        if len(chunk) < wsize:
            chunk = np.pad(chunk, (0, wsize - len(chunk)), 'constant', constant_values=0)

        # Compute the Fast Fourier Transform of the chunk
        # The first half is kept as the array is symetric
        freqspectrum = np.fft.fft(chunk)[0:1024]
        ampspectrum = np.abs(freqspectrum)
        amplitudes.append(condenseSpectrum(ampspectrum))

    return np.stack(amplitudes).astype(np.float32)
        
def preprocessAmplitudes(amplitudes: np.ndarray) -> np.ndarray:
    # Normalize amplitudes with its median values
    amplitudes = GAIN * amplitudes/np.median(amplitudes, 0)

    # Set amplitudes to zero if too small
    amplitudes[amplitudes < 0.1] = 0.0

    return amplitudes

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

def createFolder() -> None:
    os.mkdir('frames')

def createVideo() -> None:
    os.remove(f'{VIDEOPATH}')
    os.system('ffmpeg -r ' + str(FPS) + ' -f image2 -s 64x64 -i frames/%06d.png -i ' + AUDIOPATH + ' -crf 25 -vcodec libx264 -pix_fmt yuv420p ' + VIDEOPATH)
    os.rmdir('frames')

def getDevice():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    print ('Available devices ', torch.cuda.device_count())
    print ('Current cuda device ID ', torch.cuda.current_device())

    return device

if __name__ == '__main__':
    device = getDevice()
    sound = extractAudio(AUDIOPATH)
    amplitudes = stft(sound, FPS, 2048)
    processedAmplitudes = preprocessAmplitudes(amplitudes)

    rowmat = np.tile(np.linspace(-1, 1, HEIGHT), WIDTH).reshape(WIDTH, HEIGHT).T
    colmat = np.tile(np.linspace(-1, 1, WIDTH), HEIGHT).reshape(HEIGHT, WIDTH)

    rng = np.random.default_rng(seed=None)

    #cppn = buildCPPN(processedAmplitudes.shape[1], rng)
    model = CPPN(device, processedAmplitudes.shape[1], NLAYERS, HSIZE, OUTPUTSIZE)
    model.to(device)

    dataset = AudioDataset(processedAmplitudes, WIDTH, HEIGHT, ALPHA, device)
    dataloader = DataLoader(dataset, batch_size=32)

    createFolder()

    nb = 0

    with torch.no_grad():
        for data in tqdm(dataloader, desc='Generate Frames'):
            result = model(data).cpu()

            for res in result:
                frame = (255.0*res.numpy().reshape(HEIGHT, WIDTH, -1)).astype(np.uint8)
                cv2.imwrite(f'frames/{nb:06d}.png', frame)
                cv2.waitKey(1)
                nb += 1


    nbFrames = processedAmplitudes.shape[0]
    features = processedAmplitudes[0, :]




    for t in tqdm(range(nbFrames)):
        # Exponential smoothing
        features = ALPHA*features + (1-ALPHA)*processedAmplitudes[t, :]

        # Generate a frame with the features
        frame = genImage(features, cppn, rowmat, colmat)
        frame = (255.0*frame.reshape(HEIGHT, WIDTH, -1)).astype(np.uint8)

        # Show frame with opencv
        cv2.imshow('...', frame)

        # Save frame
        cv2.imwrite(f'frames/{t:06d}.png', frame)

        cv2.waitKey(1)
    
    cv2.destroyAllWindows()

    createVideo()

