import re

from typing import Dict



def parse_css_property(css_property: str) -> Dict[str, str]:

    """Parses a CSS property from a string and returns a dictionary with the property name and value.



    Args:

        css_property: A string containing a CSS property and value separated by a colon.



    Returns:

        A dictionary with the property name as the key and value as the value.

    """

    css_property = css_property.strip()

    css_property = re.sub(r"'", '"', css_property)

    words = css_property.split()

    for i, word in enumerate(words):

        if word.endswith(':'):

            property_name = word[:-1]

            value = words[i+1]



            break

    return {property_name: value}

