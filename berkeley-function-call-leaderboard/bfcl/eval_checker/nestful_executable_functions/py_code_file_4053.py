from typing import List



def generate_powers_of_two(maximum: int) -> List[int]:

    """Generates a list of integer numbers that are powers of two, starting from 1 and ending at a specified maximum value.



    Args:

        maximum: The maximum value.



    Returns:

        A list of powers of two.

    """

    exponent = 0

    powers_of_two = []

    while exponent < maximum:

        power = 2 ** exponent

        if power > maximum:

            break

        powers_of_two.append(power)

        exponent += 1

    return powers_of_two

