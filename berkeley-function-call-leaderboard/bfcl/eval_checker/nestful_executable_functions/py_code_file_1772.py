import re



def extract_text_from_p_tags(html_string: str) -> str:

    """Extracts the text inside all `p` tags from an HTML string.



    Args:

        html_string: The input HTML string.



    Returns:

        A newline-separated string containing the text inside all `p` tags.

    """

    p_tags = re.findall(r'<p>.*?</p>', html_string)

    text_inside_tags = [re.search(r'(?<=<p>).*?(?=</p>)', tag).group(0) for tag in p_tags]

    return '\n'.join(text_inside_tags)

