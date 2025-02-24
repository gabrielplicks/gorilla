from typing import Dict

import re



def extract_test_tags(docstring: str) -> Dict[str, str]:

    """Extracts test tags from a docstring.



    Args:

        docstring: The docstring to extract test tags from.



    Returns:

        A dictionary with the test tags as keys and their values as values.

    """

    tag_pattern = r"@(\S+): (\S+)"

    tags = re.findall(tag_pattern, docstring)

    return {tag: value for tag, value in tags}

