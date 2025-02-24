from typing import Union



def get_vehicle_offset_from_center(pixel_distance: Union[int, float]) -> float:

    """Calculates the offset of a vehicle from the center of the lane.



    Args:

        pixel_distance: The number of pixels the vehicle is distant from the center.



    Returns:

        The offset of the vehicle from the center of the lane in meters.

    """

    lane_width_in_meters = 3.7

    meters_per_pixel = lane_width_in_meters / 1280

    offset_in_meters = pixel_distance * meters_per_pixel

    return abs(offset_in_meters)

