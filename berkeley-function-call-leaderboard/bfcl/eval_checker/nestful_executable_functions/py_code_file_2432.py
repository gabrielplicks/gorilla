from typing import Union



def extract_stock_name(string: str) -> Union[str, None]:

    """Extracts the stock name from a string of the form `stock_name.closing_price`.



    Args:

        string: The input string.



    Returns:

        The stock name if it can be extracted, otherwise `None`.

    """

    name = string.split('.')[0]

    if len(name) > 0:

        return name

    else:

        return None

