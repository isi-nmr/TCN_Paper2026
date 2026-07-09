
import numpy as np


def estimateGhost(image):
    objectMask = np.zeros_like(image, dtype=np.bool)

    # Circle parameters
    center = (124, 128)  # (y, x) center coordinates
    radius = 80

    # Create grid of coordinates
    Y, X = np.ogrid[: objectMask.shape[0], : objectMask.shape[1]]

    # Compute distance from center
    dist_from_center = np.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)

    # Create circular mask
    mask = dist_from_center <= radius

    objectMask[mask] = True


    ghostMask = np.zeros_like(image, dtype=np.bool)

    # Circle parameters
    center = (124, 128)  # (y, x) center coordinates
    radius = 90

    # Create grid of coordinates
    Y, X = np.ogrid[: objectMask.shape[0], : objectMask.shape[1]]

    # Compute distance from center
    dist_from_center = np.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)

    # Create circular mask
    mask = dist_from_center > radius

    ghostMask[mask] = True
    ghostData = np.abs(image[ghostMask])
    imageData = np.abs(image[objectMask])

    return np.mean(ghostData) / np.mean(imageData)
