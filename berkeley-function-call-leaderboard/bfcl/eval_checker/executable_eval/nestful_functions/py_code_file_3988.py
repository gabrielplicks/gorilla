from typing import List, Dict



def generate_slack_message(events: List[Dict[str, str]]) -> List[str]:

    """Generates a Slack message for each item in a list of events.



    Args:

        events: A list of dictionaries containing event details.



    Returns:

        A list of Slack messages.

    """

    slack_messages = []



    for event in events:

        event_type = event.get("type", "")

        event_title = event.get("title", "")

        event_description = event.get("description", "")

        event_url = event.get("url", "")



        message = f"{event_type}: {event_title}"

        if event_description:

            message += f"\n{event_description}"

        message += f"\n\n[Click here]({event_url})"



        slack_messages.append(message)



    return slack_messages

