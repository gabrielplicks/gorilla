from typing import List, Union



def remove_punctuation_and_lowercase(string: str) -> str:

    """Removes punctuation from a string and converts it to lowercase.



    Args:

        string: The input string.



    Returns:

        A new string with all punctuation removed and the text converted to lowercase.

    """

    words: Union[List[str], str] = string.split() if len(string) > 0 else [string]

    no_punctuation_words: List[str] = [word.strip("!@#$%^&*()_+={}|[]\:;'<>?,./-").lower() for word in words]

    return ' '.join(no_punctuation_words)

