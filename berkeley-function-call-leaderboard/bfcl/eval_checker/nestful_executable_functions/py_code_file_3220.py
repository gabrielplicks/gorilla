from typing import List



def partition(command: str) -> List[str]:

    """Partitions a command string into subcommands.



    Args:

        command: The command string to partition.



    Returns:

        A list of subcommands.



    Raises:

        ValueError: If the command is empty or contains only whitespace characters.

    """

    subcommands = command.split()

    if not subcommands or all(s.isspace() for s in subcommands):

        raise ValueError('Invalid command: Empty or whitespace characters')



    def nested_function() -> List[str]:

        """Returns the list of subcommands."""

        return subcommands



    return nested_function

