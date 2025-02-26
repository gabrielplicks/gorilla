from datetime import datetime

from typing import Union



def tibetan_calendar_to_chinese_calendar(tibetan_date: str) -> str:

    """Converts a date in the Tibetan calendar to the same date in the Chinese calendar.



    Args:

        tibetan_date: The date in the Tibetan calendar as a string in the format `MM/DD/YYYY`.



    Returns:

        The same date in the Chinese calendar as a string in the format `YYYY-MM-DD`.

    """

    month, day, year = tibetan_date.split('/')

    chinese_date = datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d")

    chinese_date = chinese_date.strftime("%Y-%m-%d")



    return chinese_date

