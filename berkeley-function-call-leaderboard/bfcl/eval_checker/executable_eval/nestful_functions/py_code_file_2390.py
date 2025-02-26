from typing import Optional



def build_image_url(base_url: str, image_id: str, description: Optional[str] = None) -> str:

    """Builds a valid URL for an image given a base URL, an image ID, and an optional description.



    Args:

        base_url: The base URL for all images.

        image_id: The ID of the image.

        description: An optional description of the image.

    """

    url = base_url + '/images/' + image_id

    if description:

        url += f'?description={description}'

    return url

