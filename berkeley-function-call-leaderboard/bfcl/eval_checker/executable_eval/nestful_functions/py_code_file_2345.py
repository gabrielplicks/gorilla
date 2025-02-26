from typing import Dict, Tuple



def get_severity_level(score: int) -> str:

    """Maps a number between 1 and 100 to a string representing the severity level.

    Args:

        score: The number to map to a severity level.

    Returns:

        A string representing the severity level.

    """

    severity_levels: Dict[Tuple[int, int], str] = {

        (1, 30): 'low',

        (30, 60): 'medium',

        (60, 80): 'high',

        (80, 101): 'critical',

    }

    for level_range, level in severity_levels.items():

        if level_range[0] <= score <= level_range[1]:

            return level

    return 'low'  # Default severity level if score is outside the range

