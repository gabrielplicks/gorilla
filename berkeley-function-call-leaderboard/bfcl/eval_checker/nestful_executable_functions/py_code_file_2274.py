from typing import List, Tuple



def format_votes(votes: List[Tuple[int, str]]) -> List[str]:

    """Formats a list of votes into a list of strings of the format "{name}: {votes}".



    Args:

        votes: A list of tuples representing the number of votes and the name of a candidate.



    Returns:

        A list of strings of the format "{name}: {votes}".

    """

    return [f"{name}: {votes}" for votes, name in votes]

