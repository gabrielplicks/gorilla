from typing import List, Dict



def compute_weighted_average(input_list: List[Dict[str, float]]) -> float:

    """Computes the weighted average of a list of numbers and corresponding weights.

    Args:

        input_list: A list of dictionaries, each of which has three keys: `value`, `weight`, and `multiplier`.

    """

    weighted_values = []



    for dictionary in input_list:

        value = dictionary['value']

        weight = dictionary['weight']

        multiplier = dictionary['multiplier']

        weighted_value = value * weight * multiplier

        weighted_values.append(weighted_value)



    total_weighted_value = sum(weighted_values)

    total_weight = sum([d['weight'] for d in input_list])



    weighted_average = total_weighted_value / total_weight



    return weighted_average

