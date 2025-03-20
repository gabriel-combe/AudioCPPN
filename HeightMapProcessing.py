import cv2
import numpy as np
import skvideo as skv
from typing import Tuple

# Normalize between -1 and 1
def NormalizeNeg11(value):
        return ((value - np.min(value)) / (np.max(value) - np.min(value)))


# Cone Shape
def ConeHM(scale: float, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Create a meshgrid between -1 and 1
        xx, yy = np.meshgrid(np.linspace(-1, 1, kwargs['width']), np.linspace(-1, 1, kwargs['height']))
        
        # Equation that drive the morphing of the CPPN
        zz = np.sqrt(xx**2 + yy**2)

        # Normalize Z axis between -1 and 1
        zz = NormalizeNeg11(zz)

        # Scale Z axis
        zz *= scale

        return (yy[np.newaxis], xx[np.newaxis], zz[np.newaxis])


# 4 Cones Shape
def Cone4HM(scale: float, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Create a meshgrid between -1 and 1
        xx, yy = np.meshgrid(np.linspace(-1, 1, kwargs['width']), np.linspace(-1, 1, kwargs['height']))

        # Equation that drive the morphing of the CPPN
        zz = np.sqrt((xx + 0.5)**2 + (yy + 0.5)**2) \
                * np.sqrt((xx - 0.5)**2 + (yy - 0.5)**2) \
                * np.sqrt((xx - 0.5)**2 + (yy + 0.5)**2) \
                * np.sqrt((xx + 0.5)**2 + (yy - 0.5)**2)

        # Normalize Z axis between -1 and 1
        zz = NormalizeNeg11(zz)

        # Scale Z axis
        zz *= scale

        return (yy[np.newaxis], xx[np.newaxis], zz[np.newaxis])


# Video/Image Height Maps
def VideoHM(scale: float, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Create a meshgrid between -1 and 1
        xx, yy = np.meshgrid(np.linspace(-1, 1, kwargs['width']), np.linspace(-1, 1, kwargs['height']))

        # Get the heightmap image
        zz = np.squeeze(kwargs['heightmap'], axis=3)
        zz.astype(np.float16)

        # Normalize Z axis between -1 and 1
        for frame in range(zz.shape[0]):
                zz[frame] = NormalizeNeg11(zz[frame]) * scale

        # Scale Z axis
        # zz *= scale

        return (yy[np.newaxis], xx[np.newaxis], zz)