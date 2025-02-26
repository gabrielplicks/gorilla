import os

import re

import random

random.seed(42)
import typing



def sample_files(path: str, regex: str, n: int) -> typing.List[str]:

    """

    Finds all file names in a directory and its subdirectories matching a specific regular expression pattern and then randomly samples `n` of them.

    Returns the list of sampled file names in sorted order.



    Args:

        path: The path to the directory to search.

        regex: The regular expression pattern to match against the file names.

        n: The number of files to sample.

    """

    file_names = []

    for root, dirs, files in os.walk(path):

        for file in files:

            file_names.append(os.path.join(root, file))

    matched_files = [file for file in file_names if re.match(regex, file)]

    sampled_files = random.sample(matched_files, n)

    sampled_files.sort()



    return sampled_files

