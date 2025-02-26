from typing import List



def generate_markdown_unordered_list(items: List[str]) -> str:

    """Generates a Markdown-formatted unordered list from a list of strings.



    Args:

        items: A list of strings to be formatted as a Markdown unordered list.



    Returns:

        A string containing the Markdown-formatted unordered list.

    """

    markdown_list = []

    for item in items:

        markdown_list.append('* ' + item)

    return '\n'.join(markdown_list)

