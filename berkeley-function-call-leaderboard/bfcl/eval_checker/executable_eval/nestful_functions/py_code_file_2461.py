from typing import Union



def validate_image_data(img_data: Union[bytes, bytearray]) -> bool:

    """Checks if the image data is valid or not.



    Args:

        img_data: The image data to validate.



    Returns:

        True if the image data is valid, False otherwise.

    """

    if not isinstance(img_data, (bytes, bytearray)):

        return False

    if len(img_data) < 10:

        return False

    width = int.from_bytes(img_data[:2], byteorder='big', signed=False)

    height = int.from_bytes(img_data[2:4], byteorder='big', signed=False)

    return width > 0 and height > 0

