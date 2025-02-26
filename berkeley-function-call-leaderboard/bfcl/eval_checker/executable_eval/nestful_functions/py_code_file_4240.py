from typing import List



def get_paragraph_text(tags: List[str]) -> List[str]:

    """Returns a list of the text inside each paragraph tag.



    Args:

        tags: A list of html tags.



    Returns:

        A list of the text inside each paragraph tag.

    """

    text_list = []

    for tag in tags:

        if tag.startswith("<p>") and tag.endswith("</p>"):

            text = tag[3:-4]  # Slice the text inside the tag

            text_list.append(text)

    return text_list

