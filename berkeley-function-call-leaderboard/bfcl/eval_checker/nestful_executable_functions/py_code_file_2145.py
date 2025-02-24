from typing import List



def get_substrings_after_api(file_path: str) -> List[str]:

    """Returns a list of all the file substrings that appear in the file path after the last occurrence of the word 'api'.



    Args:

        file_path: The file path to extract the substrings from.



    Returns:

        A list of file substrings that appear after the last occurrence of the word 'api' in the file path.

    """

    parts = file_path.split('/')

    last_api_index = -1

    for i, part in enumerate(parts):

        if part == 'api':

            last_api_index = i

    substrings_after_api = parts[last_api_index + 1:]

    return substrings_after_api

