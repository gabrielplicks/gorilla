from typing import Union



def remove_dollar_sign_and_commas(string: str) -> Union[float, int]:

    """Removes the dollar sign and commas from a string and returns a number.



    Args:

        string: The input string.



    Returns:

        A number (float or int) without the dollar sign or commas.

    """

    return float(string.replace('$', '').replace(',', ''))

