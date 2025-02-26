from typing import Union



def ordinal_suffix(number: Union[int, float]) -> str:

    """

    Converts a given number into its ordinal representation.

    Args:

        number: A positive integer or float.

    """

    if number == 11 or number == 12 or number == 13:

        suffix = 'th'

    elif number % 10 == 1:

        suffix = 'st'

    elif number % 10 == 2:

        suffix = 'nd'

    elif number % 10 == 3:

        suffix = 'rd'

    else:

        suffix = 'th'



    return str(number) + suffix

