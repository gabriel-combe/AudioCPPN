import os
import torch

def createFolder() -> None:
    os.mkdir('frames')

def createVideo(audiopath: str, videopath: str, fps: int) -> None:
    os.remove(f'{videopath}')
    os.system('ffmpeg -r ' + str(fps) + ' -f image2 -s 64x64 -i frames/%06d.png -i ' + audiopath + ' -crf 25 -vcodec libx264 -pix_fmt yuv420p ' + videopath)
    os.rmdir('frames')

def getDevice():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    print ('Available devices ', torch.cuda.device_count())
    print ('Current cuda device ID ', torch.cuda.current_device())

    return device