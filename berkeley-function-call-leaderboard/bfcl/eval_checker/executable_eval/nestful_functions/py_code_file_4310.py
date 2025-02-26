import numpy as np



def coco_to_yolo(coco_coordinates: np.ndarray) -> np.ndarray:

    """Converts COCO coordinates into YOLO format.



    Args:

        coco_coordinates: The COCO coordinates in the form of a numpy array.



    Returns:

        The YOLO coordinates as a numpy array.

    """

    x_min = coco_coordinates[0]

    y_min = coco_coordinates[1]

    width = coco_coordinates[2]

    height = coco_coordinates[3]

    x_center = x_min + width / 2

    y_center = y_min + height / 2

    return np.array([x_center, y_center, width, height])

