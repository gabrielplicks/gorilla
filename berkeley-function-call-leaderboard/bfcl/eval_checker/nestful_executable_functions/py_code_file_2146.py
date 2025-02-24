import os

import subprocess



def tar_files(files: List[str]) -> None:

    """Creates a .tar.gz file for each string in the input list.



    Args:

        files: A list of strings representing the files to be archived.

    """

    for file in files:

        if not os.path.exists(file):

            subprocess.run(["tar", "-czvf", f"{file}.tar.gz", file])

