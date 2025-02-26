from typing import Any



def get_model_info(model: Any) -> str:

    """Returns a string in the form "Model <model_name>: <model_description>".



    Args:

        model: The model object with `.name` and `.description` attributes.

    """

    return f"Model {model.name}: {model.description}"

