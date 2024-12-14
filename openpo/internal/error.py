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


class AuthenticationError(APIError):
    """Exception raised when there are authentication issues with API keys"""

    def __init__(
        self,
        provider: str,
        message: Optional[str] = None,
        status_code: Optional[int] = None,
        response: Optional[Dict] = None,
    ):
        error_msg = (
            message if message else f"{provider} API key is invalid or not provided"
        )
        super().__init__(
            message=error_msg,
            status_code=status_code,
            response=response,
            error="authentication_error",
        )


class ProviderError(APIError):
    """Exception raised when there are provider-specific errors"""

    def __init__(
        self,
        provider: str,
        message: str,
        status_code: Optional[int] = None,
        response: Optional[Dict] = None,
    ):
        super().__init__(
            message=f"{provider} provider error: {message}",
            status_code=status_code,
            response=response,
            error="provider_error",
        )


class JSONExtractionError(Exception):
    """Base exception for JSON-related errors"""

    pass


class InvalidJSONFormatError(JSONExtractionError):
    """Exception raised when the extracted text is not valid JSON"""

    def __init__(self, message: Optional[str] = None):
        error_msg = message if message else "The extracted text is not valid JSON"
        super().__init__(error_msg)
