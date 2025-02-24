from typing import Union



def format_msisdn(msisdn: Union[str, int]) -> str:

    """Formats MSISDN numbers to the international format.



    Args:

        msisdn: The MSISDN number to format.



    Returns:

        The MSISDN number in the international format.

    """

    telephone_number = str(msisdn)[1:]

    international_format = "+2" + telephone_number



    return international_format

