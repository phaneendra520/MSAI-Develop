"""Configuration for the Azure-integrated chatbot.

MODIFIED for the "Connecting a Chatbot to an AI-as-a-Service" project:
following the assignment's DefaultConfig pattern, the Azure AI Language
key and endpoint are read from environment variables so that secrets are
never stored in source code or committed to GitHub.
"""
import os


class DefaultConfig:
    # Azure AI Language resource (Keys and Endpoint page in the Azure portal).
    API_KEY = os.environ.get("MicrosoftAPIKey", "")
    ENDPOINT_URI = os.environ.get("MicrosoftAIServiceEndpoint", "")

    # Chatbot behaviour.
    CONFIDENCE_THRESHOLD = 0.20   # lowered from 0.25 after Project 1 evaluation
    MAX_INPUT_CHARS = 500         # longer messages are rejected as malformed
    NEGATIVE_SENTIMENT_CUTOFF = 0.60

    @classmethod
    def azure_configured(cls) -> bool:
        return bool(cls.API_KEY and cls.ENDPOINT_URI)
