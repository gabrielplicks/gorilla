from typing import Union



def get_fold_id(image_id: Union[int, float]) -> int:

    """Determines which fold the image with the given identifier should be assigned to.

    Args:

        image_id: The unique identifier of the image.

    """

    return image_id % 10

