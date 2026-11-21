import os
from dotenv import load_dotenv
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import jwt

load_dotenv()

MIN_SECRET_KEY_LENGTH = 32
MIN_SECRET_KEY_DISTINCT_CHARACTERS = 8
# Fragments of the placeholders shipped in .env.example and of the usual "fill me in" values.
_PLACEHOLDER_MARKERS = ("change-me", "changeme", "change_me", "replace-me", "replace_me", "your-secret", "your_secret",
                        "your-key", "placeholder")
_PLACEHOLDER_VALUES = ("secret", "secretkey", "secret-key", "secret_key", "password", "jwt-secret", "jwt_secret")
_HOW_TO_FIX = ('Generate one with:  python -c "import secrets; print(secrets.token_hex(32))"  '
               "and set it as SECRET_KEY in .env (see .env.example).")


def validate_secret_key(value):
    """Return the JWT signing key if it passes basic sanity checks, otherwise raise RuntimeError.

    Rejected: a missing or blank value, a known placeholder (including the one in .env.example), fewer than 32
    characters, fewer than 8 distinct characters and a short pattern repeated to fill the key.
    These checks only catch obviously weak or copied values. They do NOT prove the key is random or unguessable:
    use a cryptographically random value. The error messages never contain the value.
    """
    if value is None or not value.strip():
        raise RuntimeError("SECRET_KEY is not set. It must be a random secret of at least "
                           f"{MIN_SECRET_KEY_LENGTH} characters. {_HOW_TO_FIX}")
    lowered = value.strip().lower()
    if lowered in _PLACEHOLDER_VALUES or any(marker in lowered for marker in _PLACEHOLDER_MARKERS):
        raise RuntimeError("SECRET_KEY is still a placeholder value, which anyone could use to forge login tokens. "
                           + _HOW_TO_FIX)
    if len(value) < MIN_SECRET_KEY_LENGTH:
        raise RuntimeError(f"SECRET_KEY is too short (minimum {MIN_SECRET_KEY_LENGTH} characters). " + _HOW_TO_FIX)
    if len(set(value)) < MIN_SECRET_KEY_DISTINCT_CHARACTERS or (value + value).find(value, 1) < len(value):
        raise RuntimeError("SECRET_KEY is too repetitive to be a random secret. " + _HOW_TO_FIX)
    return value


# Checked when this module is imported, i.e. before the backend creates its app, so an unsafe key stops start-up.
SECRET_KEY = validate_secret_key(os.getenv("SECRET_KEY"))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
