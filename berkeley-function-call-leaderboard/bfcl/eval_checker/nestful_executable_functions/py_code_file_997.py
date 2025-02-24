import os

from collections import Counter

from typing import Dict, List



def count_words_in_files(paths: List[str]) -> Dict[str, Dict[str, int]]:

    """Counts the occurrences of each word in a list of files.



    Args:

        paths: A list of file paths.



    Returns:

        A dictionary with the word counts for each file. The keys of the dictionary

        are the file names, and the values are dictionaries counting the number of

        occurrences of each word in the corresponding file.

    """

    counts = {}



    for path in paths:

        with open(path, 'r') as f:

            text = f.read()



        counts[os.path.basename(path)] = Counter(text.split())



    return counts

