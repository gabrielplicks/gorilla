from typing import Any



def generate_model_description(model: Any) -> str:

    """Generates a textual description of a specific model object.



    Args:

        model: The model object.



    Returns:

        A formatted string in the format "Model {name} ({id}): {description}".

    """

    name = model.name

    model_id = model.id

    description = model.description

    return f"Model {name} ({model_id}): {description}"

