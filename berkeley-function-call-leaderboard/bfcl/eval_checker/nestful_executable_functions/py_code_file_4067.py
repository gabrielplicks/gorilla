import os

import shutil

from typing import List



def generate_dataset_from_git_repository(url: str, files: List[str]) -> List[str]:

    """

    Clones a Git repository to a local directory and generates a dataset of code files.



    Args:

        url: The URL of the Git repository to clone.

        files: A list of files to copy and generate new files with the ".txt" extension.



    Returns:

        A list of the generated file paths.

    """

    os.system(f'git clone {url} <local_directory>')

    generated_files = []

    for file in files:

        shutil.copyfile(file, f'{file}.txt')

        generated_files.append(f'{file}.txt')



    return generated_files

