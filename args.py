import argparse
from HeightMapProcessing import *

heightmapfunc_dict = {
    'cone' : ConeHM,
    '4cones' : Cone4HM,
}

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
    parser.add_argument('--gain', type=float, help='Gain for normalization.')
    parser.add_argument('--gainSubBass', type=float, default=0.5, help='Gain for Sub-Bass normalization.')
    parser.add_argument('--gainBass', type=float, default=0.5, help='Gain for Bass normalization.')
    parser.add_argument('--gainLowMidrange', type=float, default=0.5, help='Gain for Low midrange normalization.')
    parser.add_argument('--gainMidrange', type=float, default=0.5, help='Gain for Midrange normalization.')
    parser.add_argument('--gainUpperMidrange', type=float, default=0.5, help='Gain for Upper midrange normalization.')
    parser.add_argument('--gainPresence', type=float, default=0.5, help='Gain for Presence normalization.')
    parser.add_argument('--gainBrillance', type=float, default=0.5, help='Gain for Brillance normalization.')
    parser.add_argument('--gainOverAudible', type=float, default=0.5, help='Gain for normalization.')
    parser.add_argument('--alpha', type=float, default=0.8, help='Alpha blending for exponential smoothing.')
    parser.add_argument('--wsize', type=int, default=1024, help='Size of the window for amplitude processing.')

    # Height map
    parser.add_argument('--scale', type=float, default=1, help='Scale of the height map.')
    parser.add_argument('--heightmap', type=str, help='Path of the height map.')
    parser.add_argument('--heightmapfunc', type=str, default='cone', choices=['cone', '4cones', 'video', 'image'], help='Function to use for the height map.')
    
    # Model hyperparameters
    parser.add_argument('--batchsize', type=int, default=32, help='Batch size for the input model.')
    parser.add_argument('--nlayers', type=int, default=8, help='Number of layer in the model.')
    parser.add_argument('--hsize', type=int, default=16, help='Size of the hidden layer.')
    parser.add_argument('--outsize', type=int, default=3, help='Size of the output.')

    return parser.parse_args()