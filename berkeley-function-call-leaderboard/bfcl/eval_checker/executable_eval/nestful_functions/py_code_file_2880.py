import random

random.seed(42)
from typing import List, Tuple



def generate_boxes(width: int, height: int, count: int) -> List[Tuple[int, int, int, int]]:

    """Generates a list of randomly generated bounding box coordinates.

    Args:

        width: The width of the image.

        height: The height of the image.

        count: The number of bounding boxes to generate.

    """

    boxes = []



    while len(boxes) < count:

        x1 = random.randint(0, width - 1)

        y1 = random.randint(0, height - 1)

        x2 = random.randint(x1, width - 1)

        y2 = random.randint(y1, height - 1)

        box = (x1, y1, x2, y2)



        if all(0 <= coord <= width - 1 for coord in box):

            boxes.append(box)



    return boxes

