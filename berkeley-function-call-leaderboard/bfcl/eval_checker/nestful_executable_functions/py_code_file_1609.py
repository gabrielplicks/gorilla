from typing import List



def is_valid_image_extension(filename: str) -> bool:

    """Checks if the file extension is a valid image extension.



    Args:

        filename: The file name to check.



    Returns:

        True if the file extension is a valid image extension, False otherwise.

    """

    valid_extensions: List[str] = ['.png', '.jpg', '.jpeg', '.gif']

    return any(filename.endswith(ext) for ext in valid_extensions)



def is_valid_image(filepath: str) -> bool:

    """Checks if the file is a valid image file.



    Args:

        filepath: The file path to check.



    Returns:

        True if the file is a valid image file, False otherwise.

    """

    return is_valid_image_extension(filepath)

