import re

from typing import Dict



def template_fill(template: str, kwargs: Dict[str, str]) -> str:

    """Fills a template with values from a dictionary using regular expressions.



    Args:

        template: A string representing a template with placeholders.

        kwargs: A dictionary of keyword arguments.



    Returns:

        The filled template as a string.

    """

    placeholders = re.findall(r"\{([\w]+)\}", template)

    for placeholder in placeholders:

        if placeholder in kwargs:

            template = re.sub(rf"\{{{placeholder}}}", kwargs[placeholder], template)



    return template

