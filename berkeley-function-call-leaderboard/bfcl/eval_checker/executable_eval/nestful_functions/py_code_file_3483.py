import re

from typing import AnyStr



def mask_phone_numbers(text: AnyStr) -> AnyStr:

    """Masks phone numbers in a given text.



    Args:

        text: The input text.



    Returns:

        The input text with phone numbers replaced by asterisks.

    """

    pattern = r'\b\d{3}-\d{3}-\d{4}\b'  # Regex pattern to match phone numbers with the format XXX-XXX-XXXX

    masked_text = re.sub(pattern, '***-***-****', text)

    return masked_text

