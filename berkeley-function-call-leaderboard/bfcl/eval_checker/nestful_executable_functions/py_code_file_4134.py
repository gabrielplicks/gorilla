import numpy as np



def pixels_match(image1: np.ndarray, image2: np.ndarray) -> bool:

    """Compares two images to find if they match exactly.



    Args:

        image1: The first image.

        image2: The second image.



    Returns:

        A boolean value indicating whether the images are identical.

    """

    image1_arr = np.array(image1)

    image2_arr = np.array(image2)



    if image1_arr.shape != image2_arr.shape:

        return False



    return np.array_equal(image1_arr, image2_arr)

