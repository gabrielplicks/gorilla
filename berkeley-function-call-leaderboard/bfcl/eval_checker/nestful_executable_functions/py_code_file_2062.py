from typing import Tuple



def generate_label(label_format_str: str, label: str) -> str:

    """Generates a label string based on the given label format string and label.



    Args:

        label_format_str: The label format string.

        label: The label to be inserted into the label format string.



    Returns:

        The generated label string.

    """

    components = label_format_str.split('::')

    label_str = ''

    for component in components:

        if component == 'LABEL':

            label_str += f"::{label}"

        else:

            label_str += f"::{component}"

    return label_str.lstrip(':')

