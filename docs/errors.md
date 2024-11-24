# Error Handling

OpenPO provides error classes to handle API and JSON-related error scenarios.

## API Error
Raised when API request to the endpoint provided is not successful.

```python
APIError(
    message: str,
    status_code: Optional[int] = None,
    response: Optional[Dict] = None,
    error: Optional[str] = None
)
```

## InvalidJSONFormatError
Raised when extracted text from model response is not valid JSON.

```python
InvalidJSONFormatError(message: Optional[str] = None)
```

!!! Note
    This error mostly raises when a model fails to follow structure given in  `response_format` and returns a non JSON response. It is recommended to implement a  retry logic to handle model inconsistency.