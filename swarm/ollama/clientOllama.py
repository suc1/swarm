import ollama

from .wrapper import (
    OllamaWrapper,
)


def GetClientOllama(base_url="http://localhost:11434"):
    try:
        ollama_client = ollama.Client(host=base_url)
        wrapped_client = OllamaWrapper(ollama_client)
        return wrapped_client
    except Exception as e:
        raise ConnectionError(
            f"Failed to connect to Ollama at {base_url}. "
            "Make sure Ollama is running and the URL is correct. "
            f"Error: {str(e)}"
        ) from e
