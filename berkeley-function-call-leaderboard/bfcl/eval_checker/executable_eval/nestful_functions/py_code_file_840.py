from typing import Dict



def handle_mapped_trait(mapped_trait: Dict[str, any]) -> Dict[str, any]:

    """

    Implements a default handler for a mapped trait using a shadow trait.

    Args:

        mapped_trait: A dict containing the trait's name and value.

    Returns:

        A dict containing the trait's name and value.

    """

    trait_name = mapped_trait['traitName']

    trait_value = mapped_trait.get('traitValue')  # Extract the trait's name and value from the mapped_trait dict



    if trait_value is None:  # Check if the trait is a mapped trait with a default value

        default_value = mapped_trait['defaultValues']['defaultValue']

        trait_value = default_value



    return {

        'traitName': trait_name,

        'traitValue': trait_value

    }

