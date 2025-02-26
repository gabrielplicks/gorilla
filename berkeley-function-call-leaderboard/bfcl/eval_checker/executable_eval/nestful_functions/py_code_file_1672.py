from typing import Union



def int_div(a: int, b: int) -> Union[int, str]:

    """Performs integer division between two integers a and b.

    If b is zero, the function returns an error message.

    Otherwise, the function returns the quotient of a divided by b,

    rounded down to the nearest integer. If the quotient is 0,

    the function returns the string "Zero".

    Args:

        a: The dividend.

        b: The divisor.

    """

    if b == 0:

        return "Error: Division by zero is undefined."



    quotient = a // b

    if quotient == 0:

        return "Zero"

    else:

        return quotient

