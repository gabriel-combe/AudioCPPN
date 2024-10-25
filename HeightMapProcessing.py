import cv2
import numpy as np
import skvideo as skv

# Normalize between -1 and 1
def NormalizeNeg11(value):
        return 2 * ((value - np.min(value)) / (np.max(value) - np.min(value))) - 1


# Cone Shape
def ConeHM(width: int, height: int, scale: float, **kwargs) -> np.ndarray:
        # Create a meshgrid between -1 and 1
        xx, yy = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
        
        # Equation that drive the morphing of the CPPN
        zz = np.sqrt(xx**2 + yy**2)

        # Normalize Z axis between -1 and 1
        zz = NormalizeNeg11(zz)

        # Scale Z axis
        zz *= scale
        return np.array([[yy, xx, zz]])


# 4 Cones Shape
def Cone4HM(width: int, height: int, scale: float, **kwargs) -> np.ndarray:
        # Create a meshgrid between -1 and 1
        xx, yy = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))

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

