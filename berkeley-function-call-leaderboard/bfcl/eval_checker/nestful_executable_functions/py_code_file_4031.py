from typing import List



def count_total_errors(error_messages: List[str]) -> int:

    """Counts the total number of errors in a list of error messages.



    Args:

        error_messages: A list of error messages.



    Returns:

        The total number of errors.

    """

    num_errors = 0



    for message in error_messages:

        if message.startswith("Error"):

            num_errors += 1

            print(message)

        elif message.startswith("Warning"):

            num_errors += 1

            print(message)



    return num_errors

