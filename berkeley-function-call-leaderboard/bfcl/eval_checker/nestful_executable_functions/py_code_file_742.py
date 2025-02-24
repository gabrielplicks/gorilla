import math



def highest_power_of_two(x: int) -> int:

    """Calculates the highest power of 2 that divides a positive integer `x`.



    The computation is performed using logarithms and then converted to an integer.



    Args:

        x: A positive integer.

    """

    log_x = math.log2(x)

    highest_power = int(log_x)

    return 2 ** highest_power

