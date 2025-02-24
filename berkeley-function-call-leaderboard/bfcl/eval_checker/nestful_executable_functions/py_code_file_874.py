from collections import Counter

from typing import List



def generate_text_document(word_list: List[str]) -> str:

    """Generates a formatted text document that displays the words in a list and the number of times they appear in the list.

    Args:

        word_list: A list of words.

    Returns:

        A formatted text document.

    """

    word_count = Counter(word_list)  # Count the occurrences of each word

    output = []

    for word, count in word_count.items():

        output.append(f"{word}\t{count}")  # Add each word and count to the output list

    output.append(f"Total Words: {len(word_list)}")  # Add the total number of words to the output list

    return "\n".join(output)  # Join the output list with newline characters

