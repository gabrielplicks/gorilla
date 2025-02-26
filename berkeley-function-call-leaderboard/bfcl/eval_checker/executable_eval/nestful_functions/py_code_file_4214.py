from typing import List



global_words: List[str] = []



def read_file_and_filter_words(file_path: str) -> List[str]:

    """Reads a text file and returns a list of words that are contained in the file and also in the list of global variable `global_words`.

    Args:

        file_path: The path to the text file.

    Returns:

        A list of words that are contained in the file and also in the list of global variable `global_words`.

    """

    try:

        with open(file_path, "r") as f:

            content = f.read()

            words = content.split()

            global_set = set(global_words)

            filtered_words = list(set(words).intersection(global_set))

            return filtered_words

    except Exception:

        pass

