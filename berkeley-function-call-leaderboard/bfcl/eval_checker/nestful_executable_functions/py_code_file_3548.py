from typing import Any, Dict, List, Union



def extract_text_values(node: Union[Dict[str, Any], List[Any], str]) -> List[str]:

    """Extracts all text values from a tree-like data structure and returns them as a single string, with each text value separated by a newline character ('\n').



    Args:

        node: The tree-like data structure to extract text values from.



    Returns:

        A list of text values.

    """

    if isinstance(node, dict):

        text_values = []

        for value in node.values():

            text_values.extend(extract_text_values(value))

        return text_values

    elif isinstance(node, list):

        text_values = []

        for item in node:

            text_values.extend(extract_text_values(item))

        return text_values

    elif isinstance(node, str):

        return [node]

    else:

        return []

