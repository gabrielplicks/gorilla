import re

import typing



def format_dataset(data: typing.Dict[str, typing.Any]) -> str:

    """

    Transforms the given dictionary into the desired format,

    where each line represents a label.



    Args:

        data: The dictionary containing the labels and their values.



    Returns:

        The formatted string containing the labels and their values.

    """

    output_format = "{key}\n{value}"

    output_lines = []



    for key, value in data.items():

        output_lines.append(output_format.format(key=key, value=value))



    output_string = "\n".join(output_lines)

    output_string = re.sub(r"\n+", "\n", output_string)



    return output_string

