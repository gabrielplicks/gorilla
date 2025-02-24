import numpy as np



def calculate_azimuth_zenith(vector: np.ndarray) -> tuple:

    """Calculates the azimuth and zenith angles of a vector in a Cartesian coordinate system.



    Args:

        vector: The vector expressed as a NumPy array.



    Returns:

        A tuple containing the azimuth and zenith angles.

    """

    azimuth = np.arctan2(vector[1], vector[0])

    xy_normal_vector = np.array([0, 0, 1])

    projection_length = np.dot(vector, xy_normal_vector)

    cos_zenith = projection_length / np.linalg.norm(vector)

    zenith = np.arccos(cos_zenith)

    return azimuth, zenith

