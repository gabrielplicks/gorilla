from typing import List, Dict



def get_events_with_duration_over(events: List[Dict], duration_threshold: float) -> List[Dict]:

    """Returns a new list of events whose duration is greater than the threshold.

    Args:

        events: A list of events.

        duration_threshold: The time duration threshold.

    """

    filtered_events = []

    for event in events:

        if event['duration'] > duration_threshold:

            filtered_events.append(event)

    return filtered_events

