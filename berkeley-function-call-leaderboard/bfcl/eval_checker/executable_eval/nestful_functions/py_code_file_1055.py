import re

from typing import List



def find_sentences_containing_keywords(text: str, keywords: List[str]) -> List[str]:

    """

    Returns a list of all sentences in the given text that contain at least one of the given keywords.

    The function handles punctuation and capitalization variations, and ignores leading and trailing whitespace.



    Args:

        text: The input text document.

        keywords: A list of keywords to search for in the sentences.

    """

    keywords = [k.lower() for k in keywords]

    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

    result = []

    for sentence in sentences:

        sentence_lower = sentence.lower()

        for keyword in keywords:

            if keyword in sentence_lower:

                result.append(sentence)

                break

    return result

