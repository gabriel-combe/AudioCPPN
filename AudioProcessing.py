import torchaudio
import numpy as np

def extractAudio(audiopath: str) -> np.ndarray:
    sound, fs = torchaudio.load(audiopath)
    print(sound.shape)
    sound = sound.numpy()[0, :]

    if fs != 44100:
        print(f'fs = {fs} [kHz]')
        print('sample rate should be 44.1 [kHZ] -> aborting ...')
        exit()
    
    return sound, fs

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

def stft(sound: np.ndarray, fs: int, fps: int, wsize: int) -> np.ndarray:
    # Length of the audio (points)
    nSamples = len(sound)

    stride = fs//fps

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
        
def preprocessAmplitudes(amplitudes: np.ndarray, gain: float) -> np.ndarray:
    # Normalize amplitudes with its median values
    amplitudes = gain * amplitudes/np.median(amplitudes, 0)

    # Set amplitudes to zero if too small
    amplitudes[amplitudes < 0.1] = 0.0

    return amplitudes