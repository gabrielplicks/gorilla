import re

import typing



def extract_license_type(license_text: str) -> str:

    """Extracts the license type from a given license text string using regular expressions.



    Args:

        license_text: The license text string to parse.



    Returns:

        A string containing the license type (for example, "GPLv3") if found,

        otherwise an empty string.

    """

    license_type_pattern = r'GPLv\d+'

    match = re.search(license_type_pattern, license_text)

    if match:

        return match.group()

    return ''

