import argparse

def get_opts():
    parser = argparse.ArgumentParser()
    
    # Input audio
    parser.add_argument('--audio', type=str,  help='Path of the audio file (mp3).')

    # Output video properties
    parser.add_argument('--width', type=int, default=512, help='Width of the output video.')
    parser.add_argument('--height', type=int, default=512, help='Height of the output video.')
    parser.add_argument('--fps', type=int, default=30, help='FPS of the output video.')
    parser.add_argument('--video', type=str,  help='Path of the output video file (mp4).')
    
    # Amplitude processing
    parser.add_argument('--gain', type=float, default=0.5, help='Gain for normalization.')
    parser.add_argument('--alpha', type=float, default=0.8, help='Alpha blending for exponential smoothing.')
    parser.add_argument('--wsize', type=int, default=1024, help='Size of the window for amplitude processing.')
    
    # Model hyperparameters
    parser.add_argument('--batchsize', type=int, default=32, help='Batch size for the input model.')
    parser.add_argument('--nlayers', type=int, default=8, help='Number of layer in the model.')
    parser.add_argument('--hsize', type=int, default=16, help='Size of the hidden layer.')
    parser.add_argument('--outsize', type=int, default=3, help='Size of the output.')

    return parser.parse_args()