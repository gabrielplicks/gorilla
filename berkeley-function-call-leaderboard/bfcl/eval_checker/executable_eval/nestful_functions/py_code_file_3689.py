from typing import List



def tokenize_sequence(sequence: str, delimiter: str) -> List[str]:

    """Tokenizes a sequence into a list of tokens, where each token is a contiguous subsequence of characters delimited by the given delimiter.



    Args:

        sequence: The input sequence to be tokenized.

        delimiter: The delimiter used to separate tokens.



    Returns:

        A list of tokens.

    """

    tokens = []

    while sequence:

        delimiter_index = sequence.find(delimiter)

        if delimiter_index != -1:

            tokens.extend(sequence[:delimiter_index].split())

            sequence = sequence[delimiter_index + len(delimiter):]

        else:

            tokens.append(sequence)

            break

    return tokens

