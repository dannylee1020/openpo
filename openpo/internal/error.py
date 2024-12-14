from typing import Dict, Optional


class APIError(Exception):
    """Base exception class for API-related errors.

    This class serves as the base for all API-related exceptions in the openpo library.
    It provides detailed error information including status codes, response data, and
    specific error messages.

    Args:
        message (str): Human-readable error message describing the issue
        status_code (Optional[int]): HTTP status code associated with the error
        response (Optional[Dict]): Raw response data from the API
        error (Optional[str]): Specific error code or identifier

    Attributes:
        message (str): The error message
        status_code (Optional[int]): The HTTP status code
        response (Optional[Dict]): The raw API response
        error (Optional[str]): The error identifier
    """

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
    """Exception raised for API authentication failures.

    This exception is raised when there are issues with API authentication, such as
    invalid API keys, expired tokens, or missing credentials.

    Args:
        provider (str): Name of the provider (e.g., 'OpenAI', 'Anthropic')
        message (Optional[str]): Custom error message. If not provided, a default
            message will be generated using the provider name
        status_code (Optional[int]): HTTP status code from the authentication attempt
        response (Optional[Dict]): Raw response data from the authentication attempt
    """

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
    """Exception raised for provider-specific errors.

    This exception is raised when a provider returns an error that is specific to
    their service, such as rate limits, invalid model names, or service-specific
    validation errors.

    Args:
        provider (str): Name of the provider (e.g., 'OpenAI', 'Anthropic')
        message (str): Detailed error message from the provider
        status_code (Optional[int]): HTTP status code from the provider
        response (Optional[Dict]): Raw response data from the provider
    """

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
    pass


class InvalidJSONFormatError(JSONExtractionError):
    """Exception raised when JSON extraction or parsing fails.

    This exception is raised when attempting to extract or parse JSON content
    that is not properly formatted or is invalid according to JSON specifications.

    Args:
        message (Optional[str]): Custom error message. If not provided, a default
            message indicating invalid JSON format will be used.
    """

    def __init__(self, message: Optional[str] = None):
        error_msg = message if message else "The extracted text is not valid JSON"
        super().__init__(error_msg)
