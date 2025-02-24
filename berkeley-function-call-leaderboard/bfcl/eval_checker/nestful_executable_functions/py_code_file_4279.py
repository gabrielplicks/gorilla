from typing import Dict, Union



greetings: Dict[str, str] = {

    "en": "Hello, {}!",

    "es": "¡Hola, {}!",

    "de": "Hallo, {}!"

}



def generate_greeting(language: str, name: str) -> str:

    """Generates a personalized greeting string in the specified language.

    Args:

        language: The language code (such as 'en' for English or 'es' for Spanish).

        name: The name to be included in the greeting.

    """

    if language in greetings:

        return greetings[language].format(name)

    return greetings["de"].format(name)

