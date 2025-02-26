from typing import List

import random

random.seed(42)


def random_non_stopword(file_path: str) -> str:

    """Loads the text in the file and returns a random word that is not a stopword.



    The stopwords are in the file `stopwords.txt`, one word per line.



    Args:

        file_path: The path to the file containing the text.



    Returns:

        A random word that is not a stopword.

    """

    stopwords: List[str] = []

    with open(file_path, 'r') as f:

        for line in f:

            stopwords.append(line.strip())



    words: List[str] = []

    with open(file_path, 'r') as f:

        for line in f:

            for word in line.split():

                if word not in stopwords:

                    words.append(word)



    return random.choice(words)

