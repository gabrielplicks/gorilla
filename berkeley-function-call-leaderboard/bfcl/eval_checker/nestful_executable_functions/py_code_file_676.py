import re

from typing import Dict, List



def parse_form(form_str: str) -> Dict[str, List[str]]:

    """Parses an HTML form in string format and extracts all the key-value pairs from it.



    The function returns a dictionary mapping the form field names to their values.

    It only supports text form fields and does not use any external libraries for parsing the HTML.



    Args:

        form_str: The HTML form in string format.



    Returns:

        A dictionary mapping the form field names to their values.

    """

    pattern = re.compile(r'<input type="text" name="(.*)" value="(.*)" />')

    matches = pattern.findall(form_str)

    result = dict()

    for key, value in matches:

        if key in result:

            result[key].append(value)

        else:

            result[key] = [value]

    return result

