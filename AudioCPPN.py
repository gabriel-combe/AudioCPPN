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
from args import get_opts

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
    # Parse arguments
    args = get_opts()

    # Initialize the CPPN parameters
    audiopath = args.audio
    videopath = args.video
    width = args.width
    height = args.height
    fps = args.fps

    # Get the device on which to run the model
    device = getDevice()

    # Get the audio data from the audio file
    sound, fs = extractAudio(audiopath)

    # Extract the amplitudes (frequencies) of the audio
    amplitudes = stft(sound, fs, fps, args.wsize)

    # Cleanup the amplitudes
    processedAmplitudes = preprocessAmplitudes(amplitudes, args.gain)

    # Create the CPPN model
    model = CPPN(device, processedAmplitudes.shape[1], args.nlayers, args.hsize, args.outsize)
    
    # Initialize weights randomly
    model.apply(init_weights)

    # Put the model on the device
    model.to(device)

    # Generate the audio dataset
    dataset = AudioDataset(processedAmplitudes, width, height, args.alpha, device)
    
    # Create a dataloader with batch size data
    dataloader = DataLoader(dataset, batch_size=args.batchsize)

    # Create frames folder
    createFolder()

    nbFrame = 0

    with torch.no_grad():
        for data in tqdm(dataloader, desc='Generate Frames'):
            result = model(data).cpu()

            for res in result:
                frame = res.numpy().reshape(height, width, -1).astype(np.uint8)
                cv2.imshow('...', frame)
                cv2.imwrite(f'frames/{nbFrame:06d}.png', frame)
                cv2.waitKey(1)
                nbFrame += 1
    
    cv2.destroyAllWindows()

    # Create a video with ffmpeg using the frame sequence
    createVideo(audiopath, videopath, fps, width, height)

