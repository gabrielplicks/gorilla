from typing import Dict



def extract_bands() -> Dict[str, str]:

    """Returns a map of operations to apply to every band in the dataset.



    Args:

        None



    Returns:

        A dictionary where the keys are the band names and the values are the operations to apply.

    """

    operations: Dict[str, str] = {}

    operations["Band 1"] = "Extract"

    operations["Band 3"] = "Extract"

    return operations

