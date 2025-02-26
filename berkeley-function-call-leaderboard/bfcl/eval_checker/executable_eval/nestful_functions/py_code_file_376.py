from typing import Union



def read_and_decode_file(filename: str, encoding: str) -> Union[str, None]:

    """Reads a text file, decodes it using the specified encoding, and returns the decoded content as a string.



    Args:

        filename: The path to the text file.

        encoding: The encoding scheme to use.



    Returns:

        The decoded content of the text file as a string. If the text file does not exist, an empty string is returned.



    Raises:

        ValueError: If the encoding scheme is not supported by the Python interpreter.

    """

    try:

        with open(filename, 'r', encoding=encoding) as f:

            return f.read()

    except FileNotFoundError:

        return ''

    except LookupError as e:

        raise ValueError('The encoding scheme is not supported by the Python interpreter.') from e

