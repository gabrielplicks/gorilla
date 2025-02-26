from typing import Tuple



def calculate_pixel_size(distance_mm: float, focal_length_mm: float) -> float:

    """Calculates the pixel size in millimeters given the physical distance and focal length of a camera.



    Args:

        distance_mm: The physical distance in millimeters.

        focal_length_mm: The focal length of the camera in millimeters.



    Raises:

        ValueError: If the distance is less than 100mm.

    """

    if distance_mm < 100:

        raise ValueError("Distance must be greater than or equal to 100mm.")

    sensor_size_mm: Tuple[float, float] = (3.2, 1.6)  # Sensor size in mm for 1920x1080 resolution

    pixel_size_mm: float = focal_length_mm * (distance_mm / sensor_size_mm[0])



    return pixel_size_mm

