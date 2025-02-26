from typing import Dict



def get_day_of_week_phrase(number: int) -> str:

    """

    Returns a phrase corresponding to the day of the week based on the input number.



    Args:

        number: An integer from 1 to 7 representing the day of the week.



    Raises:

        ValueError: If the input number is not in the range 1 to 7.

    """

    phrases: Dict[int, str] = {

        1: "Sunday",

        2: "Monday",

        3: "Tuesday",

        4: "Wednesday",

        5: "Thursday",

        6: "Friday",

        7: "Saturday"

    }



    if number not in range(1, 8):

        raise ValueError("Invalid number: number must be in range 1 to 7")



    return phrases[number]

