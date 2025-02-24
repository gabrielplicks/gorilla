from typing import Dict



def extend_trace(trace: Dict[str, str], extension: Dict[str, str]) -> Dict[str, str]:

    """Extends a partial trace with a dictionary of new attributes.



    Args:

        trace: A dictionary representing a partial trace.

        extension: A dictionary representing new attributes to add to the trace.



    Returns:

        A new dictionary combining the original trace with the new extension.

        Any duplicate attributes in the new extension will overwrite the existing ones in the original trace.

    """

    updated_trace = trace.copy()

    updated_trace.update(extension)



    return updated_trace

