from typing import List



def process_online_content(content: str) -> List[str]:

    """Processes a string by removing special characters and splitting it into a list of words.



    Args:

        content: The string to be processed.



    Returns:

        A list of cleaned words.

    """

    cleaned_content = content.replace('!', '').replace('?', '').replace(',', '').replace('.', '')

    words = cleaned_content.split(' ')

    cleaned_words = [word for word in words if word]



    return cleaned_words

