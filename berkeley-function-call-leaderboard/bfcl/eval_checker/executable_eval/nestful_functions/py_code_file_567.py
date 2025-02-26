import re

from typing import Dict



def parse_and_insert_parameters(template: str, parameters: Dict[str, str]) -> Dict[str, Dict[str, str]]:

    """Parses a template and inserts applicable parameters inside the template.



    Args:

        template: The template to parse.

        parameters: The parameters to insert into the template.



    Returns:

        A dictionary containing the parsed template and its parameters.

    """

    placeholders = re.findall("\{(\w+)\}", template)

    parsed_template = {}

    parsed_template["template"] = template

    parsed_template["parameters"] = {}

    for placeholder in placeholders:

        if placeholder in parameters:

            parsed_template["parameters"][placeholder] = parameters[placeholder]

        else:

            parsed_template["parameters"][placeholder] = "DEFAULT_VALUE"

    return parsed_template

