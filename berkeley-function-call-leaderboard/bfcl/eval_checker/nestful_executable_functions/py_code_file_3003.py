from typing import List



def join_filtered_lines(lines: List[str], sep: str = " ") -> str:

    """Joins filtered lines together with a separator character.



    Filters out any lines that are empty or contain only whitespace characters.

    Also, removes all leading and trailing whitespace from each line before joining.



    Args:

        lines: A list of lines to filter and join.

        sep: The separator character used to join the lines (default to a single space).



    Returns:

        A string with the filtered lines joined together.

    """

    return sep.join([line.strip() for line in lines if line.strip()])

