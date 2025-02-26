import re

from typing import Union



def replace_code(xml_text: str, code: Union[str, int]) -> str:

    """

    Replaces all occurrences of the `{code}` variable in a given XML text with the provided code and returns the modified XML text.



    Args:

        xml_text: The XML text to search in.

        code: The code to replace `{code}`.

    """

    return re.sub('{code}', str(code), xml_text)

