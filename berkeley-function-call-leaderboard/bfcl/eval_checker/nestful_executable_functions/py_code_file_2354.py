from typing import Dict, List



def most_common_word(text: str) -> str:

    """Returns the most common word in the given text.



    The function is case-insensitive and can handle ties.



    Args:

        text: The input text.



    Returns:

        The most common word in the text.

    """

    words: List[str] = [word.strip().lower() for word in text.split()]

    word_counts: Dict[str, int] = {}



    for word in words:

        if word in word_counts:

            word_counts[word] += 1

        else:

            word_counts[word] = 1



    max_count: int = max(word_counts.values())

    most_common_words: List[str] = [word for word, count in word_counts.items() if count == max_count]



    return most_common_words[0]

