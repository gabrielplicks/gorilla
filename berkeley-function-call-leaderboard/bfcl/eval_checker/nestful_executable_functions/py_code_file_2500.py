from typing import List, Union



def encode_data(data: List[Union[str, bytes]]) -> List[bytes]:

    """Prepares data for passing across the DLL boundary by converting the data into bytes.

    The data is a Python list of strings, each string is a word, and each word can be either ascii or utf-8.

    The function returns a list of encoded bytes, where each byte corresponds to a word in the input list.

    To encode a unicode string, the 'utf-8' encoding is used.



    Args:

        data: A list of strings or bytes.



    Returns:

        A list of bytes.

    """

    encoded_data = []

    for word in data:

        if isinstance(word, str):

            encoded_word = word.encode('utf-8')

        else:

            encoded_word = word

        encoded_data.append(encoded_word)

    return encoded_data

