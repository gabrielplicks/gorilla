from typing import List



def chunked(string: str, chunk_size: int) -> List[str]:

    """Splits a string into chunks of a specified size.



    Args:

        string: The input string to be split.

        chunk_size: The size of each chunk.



    Returns:

        A list of strings, where each string is a chunk of the input string.

    """

    chunks = []

    current_chunk = ''

    count = 0



    for char in string:

        current_chunk += char

        count += 1



        if count == chunk_size:

            chunks.append(current_chunk)

            current_chunk = ''

            count = 0



    if current_chunk:

        chunks.append(current_chunk)



    return chunks

