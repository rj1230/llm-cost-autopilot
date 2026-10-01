"""API authentication dependencies."""

import os
import secrets

from dotenv import load_dotenv
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

load_dotenv()

_bearer_scheme = HTTPBearer(
    auto_error=False,
)


def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer_scheme),  # noqa :B008
) -> None:
    """
    Require a valid Bearer API key.

    The configured key is read at request time so tests and
    application workers do not depend on import-time configuration.
    """

    api_key = os.getenv("API_KEY", "")

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is not configured.",
        )

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not secrets.compare_digest(
        credentials.credentials,
        api_key,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )
