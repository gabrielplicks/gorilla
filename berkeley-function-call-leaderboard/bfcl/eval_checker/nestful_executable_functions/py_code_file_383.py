import numpy as np



def crop_data(data: np.ndarray, crop: list[tuple[int, int]]) -> np.ndarray:

    """Crops the data based on the provided crop values and returns the cropped data.



    Args:

        data: The 3D numpy array to be cropped.

        crop: A list of three 2-element tuples, where each tuple specifies the start and end coordinates for each dimension.



    Raises:

        ValueError: If the crop values are invalid.

    """

    for start, end in crop:

        if start < 0 or end > data.shape[crop.index((start, end))]:

            raise ValueError("Invalid crop values.")

    cropped_data = data[slice(*crop[0]), slice(*crop[1]), slice(*crop[2])]



    return cropped_data

