import numpy as np



def average_intensity(image: np.ndarray) -> np.ndarray:

    """Calculates the average intensity of an RGB image.



    Args:

        image: A 3D numpy array representing the RGB image.



    Returns:

        A 2D numpy array representing the average intensity of the image.

    """

    intensity = np.mean(image, axis=2)

    return intensity

