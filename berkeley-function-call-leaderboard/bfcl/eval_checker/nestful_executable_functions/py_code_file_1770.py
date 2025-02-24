from datetime import datetime, timedelta

from typing import Union



def convert_to_pst(utc_time_str: str) -> str:

    """Converts a date and time string in UTC to the corresponding date and time in Pacific Standard Time (PST).

    Args:

        utc_time_str: The date and time string in UTC. The format is "YYYY-MM-DD HH:MM:SS".

    Returns:

        The date and time string in PST. The format is "YYYY-MM-DD HH:MM:SS".

    """

    utc_time = datetime.strptime(utc_time_str, "%Y-%m-%d %H:%M:%S")

    pst_time = utc_time - timedelta(hours=8)

    return pst_time.strftime("%Y-%m-%d %H:%M:%S")

