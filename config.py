# config.py
import os
from typing import Optional

class Config:
    """
    Configuration class to access environment variables.
    """
    # Define environment variables with default values in config class
    DEFAULT_LANGUAGE: str = os.getenv("DEFAULT_LANGUAGE", "en")
    LANGCHAIN_TEMPERATURE: float = float(os.getenv("LANGCHAIN_TEMPERATURE", 0.1))
    OPENAI_DEPLOYMENT_NAME: Optional[str] = os.getenv("OPENAI_DEPLOYMENT_NAME")
    OPENAI_API_VERSION: Optional[str] = os.getenv("OPENAI_API_VERSION")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT: Optional[str] = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_KEY: Optional[str] = os.getenv("AZURE_OPENAI_KEY")
    MAX_RESULTS: int = int(os.getenv("MAX_RESULTS", 2))
    TAVILY_API_KEY: Optional[str] = os.getenv("TAVILY_API_KEY")
    AUTH_USERNAME: Optional[str] = os.getenv("AUTH_USERNAME")
    AUTH_PASSWORD: Optional[str] = os.getenv("AUTH_PASSWORD")



    def __init__(self):
        # Optional: Validate required environment variables
        required_vars = [
            "OPENAI_DEPLOYMENT_NAME",
            "OPENAI_API_VERSION",
            "OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_KEY",
            "TAVILY_API_KEY",
            "AUTH_USERNAME",
            "AUTH_PASSWORD"
           
           
        ]
        for var in required_vars:
            value = getattr(self, var)
            if value is None:
                raise ValueError(f"Required environment variable {var} is missing.")

# Create an instance of the Config class to be used in other files
config = Config()