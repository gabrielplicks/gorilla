from typing import Dict



def format_stats(stats: Dict[str, float], format_str: str) -> str:

    """Formats statistics using a format string.

    Args:

        stats: A dictionary of statistics with keys as `mean`, `median`, or `standard_deviation`.

        format_str: A string with replacement field names for the keys in `stats`.

    """

    return format_str.format(**stats)

