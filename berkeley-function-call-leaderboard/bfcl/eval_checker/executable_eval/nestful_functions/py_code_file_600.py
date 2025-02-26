import json

from typing import List, Dict



def load_image_json(path: str) -> List[List[tuple]]:

    """Loads the data from a JSON file into an image object where each line represents a row of pixels in the image.



    Args:

        path: The path to the JSON file.



    Returns:

        A list of lists representing the image, where each inner list represents a row of pixels and each pixel is represented as a tuple of integers.

    """

    with open(path, 'r') as file:

        json_data = file.read()

    image_data = json.loads(json_data)

    image = []

    for row in image_data:

        processed_row = []

        for pixel in row:

            rgb = (int(pixel['r']), int(pixel['g']), int(pixel['b']))

            processed_row.append(rgb)

        image.append(processed_row)

    return image

