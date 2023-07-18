import os
from typing import List
from dotenv import load_dotenv
from os.path import exists, isfile 
import json

load_dotenv()

class Configuration:
    """
    Class responsible for handling the configuration of services
    """

    def __init__(self) -> None:
        self._assert_environment_variables()

    def _assert_environment_variables(self) -> None:
        """
        Verify that all the required environment variables are defined.
        """
        required_variables: dict = [
            "GENAI_KEY",
            "GENAI_ENDPOINT"
        ]
        for required_variable in required_variables:
            assert required_variable in os.environ, f"The environment variable {required_variable} is missing."

    @property
    def genai_api_key(self) -> str:
        """
        Returns the api key for BAM
        """
        return os.environ["GENAI_KEY"]

    @property
    def genai_endpoint(self) -> str:
        """
        Returns the api key for BAM
        """
        return os.environ["GENAI_ENDPOINT"]
    