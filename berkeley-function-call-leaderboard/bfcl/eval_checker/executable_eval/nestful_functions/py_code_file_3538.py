from typing import List



def filter_product_labels(labels: List[str], identifier: str) -> List[str]:

    """Filters a list of product labels by a certain identifier.



    Args:

        labels: A list of product labels.

        identifier: A label or a list of labels.



    Returns:

        A new list of product labels that contain the identifier.

    """

    return [label for label in labels if identifier in label]

