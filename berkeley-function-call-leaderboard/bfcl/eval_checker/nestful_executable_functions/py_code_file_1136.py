import re

from datetime import datetime

from typing import Callable



CUSTOM_FORMATTERS: dict[str, Callable[[datetime], str]] = {

    '%Z': lambda dt: 'UTC'

}



def format_utc_datetime_with_custom_format(dt: datetime, format_string: str) -> str:

    """Formats a UTC datetime object using a custom format string.



    Args:

        dt: The datetime object to format.

        format_string: The custom format string.

    """

    format_string = re.sub(r'%Z', 'UTC', format_string)

    for code in re.findall(r'%[a-zA-Z]', format_string):

        if code not in CUSTOM_FORMATTERS:

            continue

        format_string = format_string.replace(code, CUSTOM_FORMATTERS[code](dt))



    return dt.strftime(format_string)

