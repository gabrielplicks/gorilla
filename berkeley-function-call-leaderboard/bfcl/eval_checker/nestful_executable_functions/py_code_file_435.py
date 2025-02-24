from typing import Union



def get_ordinal_number(num: Union[int, float]) -> Union[str, None]:

    """Generates ordinal numbers as strings.



    Args:

        num: A positive integer between 1 and 100.



    Returns:

        The ordinal number of the input as a string, or None if the input is invalid.

    """

    if not isinstance(num, int) or num < 1 or num > 100:

        return None

    if num in (11, 12, 13):

        return f"{num}th"



    last_digit = num % 10

    if last_digit == 1:

        return f"{num}st"

    elif last_digit == 2:

        return f"{num}nd"

    elif last_digit == 3:

        return f"{num}rd"

    else:

        return f"{num}th"

