from typing import List, Dict

import json



def collect_images(data: str, keyword: str) -> List[Dict]:

    """Collects all images whose name contains the given keyword and sorts them by their id in descending order.



    Args:

        data: A JSON string containing image data.

        keyword: The keyword to search for in image names.



    Returns:

        A list of image dictionaries.

    """

    json_data = json.loads(data)

    images = [image for image in json_data["images"] if keyword in image["name"]]

    images.sort(key=lambda image: image["id"], reverse=True)



    return images

