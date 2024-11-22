from typing import Dict, Optional


class APIError(Exception):
    """Custom exception for API-related errors"""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response: Optional[Dict] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class JSONExtractionError(Exception):
    pass


class InvalidJSONFormatError(JSONExtractionError):
    "Raised when the extracted text is not valid JSON"

    def __init__(self, message=None):
        super().__init__(message)
