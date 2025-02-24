from typing import Tuple



def bin_ops(x: int, y: int) -> Tuple[int, int, int]:

    """Performs bitwise operations on two integers and returns the results.



    Args:

        x: The first integer.

        y: The second integer.



    Returns:

        A tuple containing the results of the `AND`, `OR`, and `XOR` operations.

    """

    x_bin = format(x, 'b')

    y_bin = format(y, 'b')

    and_result = int(x_bin, 2) & int(y_bin, 2)

    or_result = int(x_bin, 2) | int(y_bin, 2)

    xor_result = int(x_bin, 2) ^ int(y_bin, 2)



    return and_result, or_result, xor_result

