from typing import List



def random_variables_notation(random_variables: List[str]) -> str:

    """Calculates the notation for the joint distribution of a list of random variables.



    Args:

        random_variables: A list of strings representing the random variables.



    Returns:

        A string representing the notation for the joint distribution of the input random variables.

    """

    if len(random_variables) == 1:

        return f'P({random_variables[0]})'

    else:

        return 'P(' + ', '.join(random_variables) + ')'

