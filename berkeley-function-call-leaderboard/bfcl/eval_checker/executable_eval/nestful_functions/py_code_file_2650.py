from typing import Tuple



def quarter_steps_to_seconds(quarter_steps: int, tempo: int) -> float:

    """Converts quarter steps to seconds given the tempo in quarters per minute (QPM).

    Args:

        quarter_steps: An integer representing the number of quarter steps to convert.

        tempo: An integer representing the tempo in QPM.

    """

    quarter_duration = 60.0 / tempo

    return quarter_steps * quarter_duration

