from typing import List



def get_camera_coordinates(camera_name: str) -> List[int]:

    """Returns a list of two numbers representing the camera's x and y coordinates.



    Args:

        camera_name: A string representing the camera name. The camera name will always have the format `CAM_X_Y` where `X` and `Y` are integers of the camera's x and y coordinates.



    Raises:

        ValueError: If the camera name format is invalid.

    """

    components = camera_name.split('_')

    if len(components) != 3 or components[0] != 'CAM':

        raise ValueError('Invalid camera name format')

    x = int(components[1])

    y = int(components[2])

    return [x, y]

