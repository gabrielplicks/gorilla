from typing import List, Dict



def filter_events_by_severity(events: List[Dict], threshold: int) -> List[Dict]:

    """Filters a list of events based on a severity threshold.



    Args:

        events: A list of dictionaries representing events. Each dictionary should have a 'severity' key that represents its severity level.

        threshold: The severity threshold provided by the user.



    Returns:

        A filtered list of events whose severity is below or equal to the threshold.

    """

    filtered_events = []

    for event in events:

        if event['severity'] <= threshold:

            filtered_events.append(event)

    return filtered_events

