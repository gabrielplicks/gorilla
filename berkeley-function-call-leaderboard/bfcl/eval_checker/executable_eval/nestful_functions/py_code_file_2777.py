import json



def parse_json_log_line(log_line: str) -> dict:

    """Parses a log line in JSON format and returns a Python object.

    If the log line is not valid JSON, returns None.

    Args:

        log_line: The log line to parse.

    """

    try:

        data = json.loads(log_line)

        return data

    except json.JSONDecodeError:

        return None

