import glob

import re

from typing import List



def count_text_files(directory: str) -> int:

    """Counts the number of text files in a given directory with names in the format `some_name.txt`.



    Args:

        directory: The directory to search for text files.



    Returns:

        The number of text files in the directory with names in the format `some_name.txt`.

    """

    pattern = re.compile(r'^\w+\.txt$')

    return len([f for f in glob.glob(f'{directory}/*.txt') if pattern.match(f.split('/')[-1])])

