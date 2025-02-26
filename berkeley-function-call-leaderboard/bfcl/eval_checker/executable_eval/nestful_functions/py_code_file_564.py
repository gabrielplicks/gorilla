from typing import Union



def currency_string_to_numeric(currency_string: str) -> Union[tuple, None]:

    """Converts a string-formatted amount of currency to a numeric value.

    The function supports the following currency symbols: '$', '€', and '¥'.

    If the string format is not a valid currency representation, the function returns `None`.



    Args:

        currency_string: The string-formatted currency value.



    Returns:

        A tuple containing the numeric value and the currency symbol, or `None`.

    """

    currency_symbols = {'$': 'USD', '€': 'EUR', '¥': 'JPY'}

    for symbol in currency_symbols:

        if currency_string.endswith(symbol):

            amount = float(currency_string.rstrip(symbol))

            return amount, currency_symbols[symbol]

    return None

