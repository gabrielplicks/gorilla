from typing import Dict



def calculate_price_to_weight_ratio(input_dict: Dict[str, Dict[str, float]]) -> Dict[str, float]:

    """Calculates the price to weight ratio for each item in a dictionary.



    Args:

        input_dict: A dictionary where the keys are item names and the values are dictionaries containing the price and weight for each item.



    Returns:

        A dictionary where the keys are item names and the values are the price to weight ratios.

    """

    output_dict = {}

    for key, value in input_dict.items():

        price = value['price']

        weight = value['weight']

        output_dict[key] = price / weight

    return output_dict

