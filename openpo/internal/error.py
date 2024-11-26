from typing import Dict, Optional


class APIError(Exception):
    """Exception for API-related errors with detailed error information"""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response: Optional[Dict] = None,
        error: Optional[str] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.response = response
        self.error = error

        super().__init__(message)


class JSONExtractionError(Exception):
    """Base exception for JSON-related errors"""

    pass


class InvalidJSONFormatError(JSONExtractionError):
    """Exception raised when the extracted text is not valid JSON"""

    def __init__(self, message: Optional[str] = None):
        error_msg = message if message else "The extracted text is not valid JSON"
        super().__init__(error_msg)


class InvalidStreamError(Exception):

    def __init__(self, message: Optional[str] = None):
        error_msg = message if message else "data not returned from stream"
