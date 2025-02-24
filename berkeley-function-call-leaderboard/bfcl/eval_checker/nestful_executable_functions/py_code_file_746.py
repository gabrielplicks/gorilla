from typing import Dict, List



def open_boxes(boxes: Dict[str, str]) -> List[str]:

    """Simulates the opening of boxes, each containing a key to open the next box.

    Args:

        boxes: A dictionary that represents the box opening process.

    Returns:

        A list of boxes that were successfully opened.

    """

    opened_boxes = []



    current_box = 1  # Start with box 1

    while current_box in boxes:

        opened_boxes.append(current_box)

        current_box = boxes[current_box]



    return opened_boxes

