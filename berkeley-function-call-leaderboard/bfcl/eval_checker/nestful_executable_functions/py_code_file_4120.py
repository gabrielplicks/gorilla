from typing import Optional



def get_image_class_from_path(file_path: str) -> Optional[str]:

    """Returns the corresponding image class (e.g., "cat" or "dog") from a file path.



    Args:

        file_path: The file path of the image.



    Returns:

        The image class if the file name ends with ".jpg" and contains "cat." or "dog.",

        otherwise None.

    """

    file_name = file_path.split("/")[-1]  # Extract the file name from the file path

    if file_name.endswith(".jpg") and "cat." in file_name:

        return "cat"

    elif file_name.endswith(".jpg") and "dog." in file_name:

        return "dog"

    else:

        return None

