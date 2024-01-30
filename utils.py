import os
import torch
import shutil

def createFolder() -> None:
    if os.path.exists('frames'):
        shutil.rmtree('frames', ignore_errors=True)
    
    os.mkdir('frames')

def createVideo(audiopath: str, videopath: str, fps: int, width: int, height: int) -> None:
    try:
        os.remove(f'{videopath}')
    except FileNotFoundError:
        print("File is not present in the system.")

    os.system(f'ffmpeg -r {str(fps)} -f image2 -s {width}x{height} -i frames/%06d.png -i {audiopath} -crf 25 -vcodec libx264 -pix_fmt yuv420p {videopath}')
    shutil.rmtree('frames', ignore_errors=True)

def getDevice():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    print ('Available devices ', torch.cuda.device_count())
    print ('Current cuda device ID ', torch.cuda.current_device())

    return device