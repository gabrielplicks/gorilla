import json



def parse_context_params(context_str: str) -> dict:

    """Parses the context parameters of a lambda function.



    Args:

        context_str: A string containing the context parameters of a lambda function.



    Returns:

        A dictionary containing the context parameters of the lambda function.

    """

    context_dict = json.loads(context_str)

    context_dict["memory_limit_in_mb"] = int(context_dict["memory_limit_in_mb"])

    context_dict["deadline_ms"] = int(context_dict["deadline_ms"])



    return context_dict

