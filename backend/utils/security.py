"""Password hashing (scrypt) and JWT helpers."""

import base64
import hashlib
import hmac
import os
import time

import jwt

# Deliberately NOT in Config: the stored hash (scrypt$<salt>$<hash>) does not record these
# parameters, so changing them would make every existing password fail to verify.
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    """Return ``scrypt$<salt>$<hash>`` (urlsafe base64) for a password."""
    salt = os.urandom(16)
    digest = hashlib.scrypt(
        password.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=32
    )
    b64 = base64.urlsafe_b64encode
    return f"scrypt${b64(salt).decode()}${b64(digest).decode()}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time check of a password against a ``hash_password`` value."""
    try:
        scheme, salt_b64, hash_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        salt = base64.urlsafe_b64decode(salt_b64)
        expected = base64.urlsafe_b64decode(hash_b64)
    except ValueError:
        return False
    actual = hashlib.scrypt(
        password.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=len(expected)
    )
    return hmac.compare_digest(actual, expected)


def create_token(user_id: int, secret: str, expires_minutes: int) -> str:
    now = int(time.time())
    payload = {"sub": str(user_id), "iat": now, "exp": now + expires_minutes * 60}
    return jwt.encode(payload, secret, algorithm=_ALGORITHM)


def decode_token(token: str, secret: str) -> int | None:
    """Return the user id from a valid token, or None if invalid/expired."""
    try:
        payload = jwt.decode(token, secret, algorithms=[_ALGORITHM], options={"require": ["exp"]})
        return int(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        return None
