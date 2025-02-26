from typing import List



class Slog:

    def __init__(self, timestamp: str, message: str, level: str):

        self.timestamp = timestamp

        self.message = message

        self.level = level



def format_slogs(slog_objects: List[Slog]) -> str:

    """Formats a list of slog objects into a string.



    Each line in the output string is in the format: "{timestamp} [{level}] {message}".



    Args:

        slog_objects: A list of slog objects.



    Returns:

        A formatted string containing the slog lines.

    """

    formatted_slogs = []



    for slog in slog_objects:

        line = f"{slog.timestamp} [{slog.level}] {slog.message}"

        formatted_slogs.append(line)

    return '\n'.join(formatted_slogs)

