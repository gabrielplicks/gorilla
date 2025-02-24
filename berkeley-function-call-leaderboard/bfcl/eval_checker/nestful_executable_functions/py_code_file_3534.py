from typing import Union



def get_calibration_steps(instrument_id: str) -> Union[int, None]:

    """Returns the number of calibration steps necessary for a particular instrument.

    Args:

        instrument_id: A string that uniquely identifies the instrument.

    """

    if instrument_id.startswith('X'):

        return 1

    elif instrument_id.startswith('Y'):

        return 2

    elif instrument_id.startswith('Z'):

        return 3

    else:

        return 0

