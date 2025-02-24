from typing import List



def extend_image_urls(images: List[str], base_path: str) -> List[str]:

    """Extends a list of image URLs with a base URL path.



    Args:

        images: A list of image URLs.

        base_path: The base URL path to add to the image URLs.



    Returns:

        A new list of URLs with the base URL path added.

    """

    extended_images = []

    for image in images:

        if not image.startswith("http://") and not image.startswith("https://"):

            image = base_path + image

        extended_images.append(image)

    return extended_images

