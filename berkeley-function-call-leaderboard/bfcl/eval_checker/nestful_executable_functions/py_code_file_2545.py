import re



def format_text(template_string: str, data: dict) -> str:

    """

    Formats a template string by replacing keys with corresponding values from a dictionary.

    Args:

        template_string: The template string to format.

        data: A dictionary containing key-value pairs.

    """

    key_pattern = re.compile(r'\${(\w+)}')

    def replace_key(match):

        key = match.group(1)

        if key in data:

            return str(data[key])

        else:

            return match.group(0)

    return key_pattern.sub(replace_key, template_string)

