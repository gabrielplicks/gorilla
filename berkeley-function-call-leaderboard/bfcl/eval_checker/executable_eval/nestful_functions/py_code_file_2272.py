import numpy as np



def calculate_toa_flux(flux: np.ndarray, cos_dir: np.ndarray) -> float:

    """Calculates the radiative flux TOA using the given equation.



    Args:

        flux: The radiative flux vector of shape (N, 1).

        cos_dir: The direction cosine vector of shape (N, 1).

    """

    return np.dot(np.sqrt(np.abs(flux)), np.sqrt(np.abs(cos_dir))) / np.pi

