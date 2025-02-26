from typing import List



def consonants_only(text: str) -> str:

    """Extracts consonants from a given text string without using the `re` module.



    Args:

        text: The input text string.



    Returns:

        A new string with only consonants.

    """

    VOWELS = "aeiou"

    consonants = ""



    for char in text:

        if char not in VOWELS:

            consonants += char



    return consonants

