from typing import List



def pad_array(array: List[float], number: float, length: int) -> List[float]:

    """Pads an array of numbers with a single number to a specific length, rounding the numbers to the nearest tenth.



    Args:

        array: The array of numbers to pad.

        number: The number to use for padding.

        length: The desired length of the padded array.

    """

    rounded_array = [round(number, 1) for number in array]

    diff = length - len(rounded_array)

    if diff > 0:

        rounded_array.extend([round(number, 1)] * diff)



    return rounded_array

