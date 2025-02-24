import re



def remove_markdown_links(text: str) -> str:

    """Removes Markdown links from a given Markdown document.



    Args:

        text: The Markdown document as a string.



    Returns:

        A new string with all Markdown links removed.

    """

    assert isinstance(text, str), "Input must be a string"



    markdown_links_pattern = r'\[.*?\]\(.*?\)'

    text_without_links = re.sub(markdown_links_pattern, '', text)



    return text_without_links

