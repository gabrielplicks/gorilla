import numpy as np



def augment_image(image: np.ndarray, prob: float) -> np.ndarray:

    """Augments an image by flipping it horizontally randomly with a given probability.



    Args:

        image: The input image.

        prob: The probability of flipping the image horizontally.



    Raises:

        TypeError: If the image is not a numpy array.

        ValueError: If the probability is not a float between 0 and 1.

    """

    if not isinstance(image, np.ndarray):

        raise TypeError('Image must be a numpy array')

    if not isinstance(prob, float) or not 0 <= prob <= 1:

        raise ValueError('Probability must be a float between 0 and 1')



    if np.random.random() < prob:

        return np.fliplr(image)

    else:

        return image

