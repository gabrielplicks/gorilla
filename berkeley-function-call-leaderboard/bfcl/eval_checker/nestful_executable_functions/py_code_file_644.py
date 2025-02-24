from typing import List



def generate_bin_nums(size: int, padding: int) -> List[str]:

    """Generates a list of binary numbers with the specified size and padding length.



    Args:

        size: The number of binary numbers to generate.

        padding: The length of each binary number.



    Returns:

        A list of binary numbers, each with the specified padding length.

    """

    return [bin(i).replace('0b', '').zfill(padding) for i in range(1, size+1)]

