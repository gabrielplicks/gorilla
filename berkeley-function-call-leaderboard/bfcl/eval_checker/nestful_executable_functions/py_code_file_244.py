from typing import Dict



def symbol_frequencies(stream: List[int]) -> Dict[str, int]:

    """Calculates the frequencies of symbols in a stream of bytes.



    Args:

        stream: A list of bytes representing the stream.



    Returns:

        A dictionary where the keys are the symbols and the values are the corresponding frequencies.

    """

    frequencies = {}

    for byte in stream:

        symbol = chr(byte)  # Convert the byte to a symbol

        if symbol not in frequencies:

            frequencies[symbol] = 0

        frequencies[symbol] += 1

    return frequencies

