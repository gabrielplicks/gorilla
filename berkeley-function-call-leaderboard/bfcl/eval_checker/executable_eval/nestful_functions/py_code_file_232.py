from datetime import timedelta

from string import Template



def format_delta(delta: timedelta) -> str:

    """Formats a time delta into a human-readable string.



    Args:

        delta: The time delta to format.

    """

    template_str = Template("$days days, $hours hours, $minutes minutes")

    days = delta.days

    hours, remainder = divmod(delta.seconds, 3600)

    minutes, _ = divmod(remainder, 60)

    output = template_str.substitute(days=days, hours=hours, minutes=minutes)



    return output

