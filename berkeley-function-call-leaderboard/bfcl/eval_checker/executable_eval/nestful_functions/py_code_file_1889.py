import codecs

from typing import List



def read_list(path: str, encoding: str) -> List[str]:

    """Reads the contents of a text file and returns a list of non-empty lines.



    Args:

        path: The file path.

        encoding: The text encoding.



    Returns:

        A list of non-empty lines.

    """

    assert path, "path must be a non-empty string"

    assert encoding in codecs.encodings.aliases.aliases, f"{encoding} is not a valid encoding"



    with codecs.open(path, "r", encoding) as file:

        lines = file.read().splitlines()



    return [line for line in lines if line]

