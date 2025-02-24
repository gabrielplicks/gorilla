from typing import List



def capitalize_sentence(sentence: str, capitalize_first: bool = True) -> str:

    """Capitalizes each word in a sentence.



    Args:

        sentence: The sentence to capitalize.

        capitalize_first: Whether to capitalize the first word. Default is True.



    Returns:

        The capitalized sentence.

    """

    words: List[str] = sentence.split(' ')

    capitalized_words: List[str] = [word.title() if capitalize_first or i > 0 else word for i, word in enumerate(words)]

    capitalized_sentence: str = ' '.join(capitalized_words)

    return capitalized_sentence

