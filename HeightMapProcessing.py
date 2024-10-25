import cv2
import numpy as np
import skvideo as skv

# Normalize between -1 and 1
def NormalizeNeg11(value):
        print(np.min(value))
        print(np.max(value))
        return ((value - np.min(value)) / (np.max(value) - np.min(value)))


# Cone Shape
def ConeHM(scale: float, **kwargs) -> np.ndarray:
        # Create a meshgrid between -1 and 1
        xx, yy = np.meshgrid(np.linspace(-1, 1, kwargs['width']), np.linspace(-1, 1, kwargs['height']))
        
        # Equation that drive the morphing of the CPPN
        zz = np.sqrt(xx**2 + yy**2)

        # Normalize Z axis between -1 and 1
        zz = NormalizeNeg11(zz)

        # Scale Z axis
        zz *= scale
        return np.array([[yy, xx, zz]])


# 4 Cones Shape
def Cone4HM(scale: float, **kwargs) -> np.ndarray:
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
        return np.array([[yy, xx, zz]])


# Video Height Maps
def VideoHM(scale: float, **kwargs) -> np.ndarray:
        # Create a meshgrid between -1 and 1
        xx, yy = np.meshgrid(np.linspace(-1, 1, kwargs['width']), np.linspace(-1, 1, kwargs['height']))

        # Get the heightmap image
        zz = np.squeeze(kwargs['heightmap'], axis=3)
        zz.astype(np.float16)

        resultMorph = []

        for frame in range(zz.shape[0]):
                resultMorph.append(np.array([
                        yy,
                        xx,
                        NormalizeNeg11(zz[frame]) * scale
                ]))

        print(resultMorph.shape)
        print(resultMorph)

        return np.array(resultMorph)