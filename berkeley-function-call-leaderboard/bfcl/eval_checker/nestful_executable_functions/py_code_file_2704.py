import re

from typing import Dict



def template_tag(template: str, values: Dict[str, str]) -> str:

    """

    Fills a template string with named values.



    The template string contains template tags enclosed in double curly brackets,

    e.g. `Hello, {{name}}!`. The function replaces the template tags with the

    corresponding named values.



    Args:

        template: The template string.

        values: A dictionary of named values.

    """

    pattern = re.compile(r'\{\{(.*?)\}\}')

    def replace_with_value(match):

        return str(values.get(match.group(1), ''))

    return pattern.sub(replace_with_value, template)

