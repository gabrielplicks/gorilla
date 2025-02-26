from typing import List, Union



def sanitize_list(list_of_sequences: List[List[Union[str, bytes]]], max_length: int) -> List[List[Union[str, bytes]]]:

    """Truncates each sequence in a list of sequences to a maximum length.



    Args:

        list_of_sequences: A list of sequences, where each sequence is a list of strings or bytes.

        max_length: The maximum length of each sequence.



    Returns:

        A list of sequences, where each sequence is truncated to the maximum length.

    """

    sanitized_list = []

    for sequence in list_of_sequences:

        sanitized_sequence = []

        for i in range(max_length):

            sanitized_sequence.append(sequence[i])

        sanitized_list.append(sanitized_sequence)

    return sanitized_list

