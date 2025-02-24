import re

import datetime

from typing import Optional



def parse_date(input_string: str) -> Optional[datetime.date]:

    """Parses a string input into a datetime object if the input matches the format "YYYYMMDD".



    Args:

        input_string: The string input to be parsed.



    Returns:

        A datetime object if the input matches the format "YYYYMMDD", otherwise None.

    """

    if re.match(r"^\d{8}$", input_string):

        return datetime.datetime.strptime(input_string, "%Y%m%d").date()

    else:

        return None

