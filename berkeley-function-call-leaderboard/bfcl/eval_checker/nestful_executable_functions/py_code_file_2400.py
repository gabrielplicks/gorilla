from typing import List



def read_file_and_output_words_longer_than_5() -> List[str]:

    """Reads a file called `test.txt` and outputs a list of words which are longer than 5 letters.



    Returns:

        A list of words which are longer than 5 letters.

    """

    with open('test.txt', 'r') as f:

        contents = f.read()

        words = contents.split()

        filtered_words = list(filter(lambda x: len(x) > 5, words))

        return filtered_words

