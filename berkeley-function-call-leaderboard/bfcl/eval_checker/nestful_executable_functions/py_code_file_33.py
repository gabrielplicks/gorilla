from typing import List



def conversion_rate(events: List[int], event_status: List[int]) -> float:

    """

    Calculates the conversion rate of a stream of events. If the number of events is zero, returns None.

    Args:

        events: A list of events.

        event_status: A list of event statuses.

    """

    try:

        return sum(event_status) / len(events)

    except ZeroDivisionError:

        return None

