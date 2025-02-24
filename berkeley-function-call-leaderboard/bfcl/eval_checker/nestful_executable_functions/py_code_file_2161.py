from typing import List, Union



def scale_vector(vector: List[Union[int, float]], factor: Union[int, float]) -> List[Union[int, float]]:

    """Scales a given vector by a given constant factor.



    Args:

        vector: A list of numbers representing the vector.

        factor: The scaling factor to apply to the vector components.



    Raises:

        ValueError: If the input vector is not a valid list of numbers.

    """

    if not isinstance(vector, list):

        raise ValueError("Input vector must be a list of numbers.")



    scaled_vector = []

    for component in vector:

        if not isinstance(component, (int, float)):

            raise ValueError("Each vector component must be a number.")

        scaled_vector.append(component * factor)



    return scaled_vector

