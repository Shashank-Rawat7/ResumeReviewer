import logging
import os
import time
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

# PostgreSQL connection URL
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")

# An explicit DATABASE_URL wins. Otherwise the URL is built from the parts with URL.create, which takes the raw
# password (characters such as @ : / % # need no manual escaping).
DATABASE_URL = os.getenv("DATABASE_URL") or URL.create(
    "postgresql+psycopg2",
    username=POSTGRES_USER,
    password=POSTGRES_PASSWORD,
    host=POSTGRES_HOST,
    port=int(POSTGRES_PORT) if POSTGRES_PORT else None,
    database=POSTGRES_DB,
)
print("Database connection URL configured.")

# Create the engine
engine = create_engine(DATABASE_URL)

# Session and Base
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# PostgreSQL may still be starting (or restarting) when the backend starts, so table creation is retried.
DB_STARTUP_ATTEMPTS = 30
DB_STARTUP_DELAY_SECONDS = 2


def init_db(attempts=DB_STARTUP_ATTEMPTS, delay=DB_STARTUP_DELAY_SECONDS):
    """Create all tables if they don't exist, waiting for the database to accept connections."""
    for attempt in range(1, attempts + 1):
        try:
            Base.metadata.create_all(bind=engine)
            return
        except OperationalError as exc:
            if attempt == attempts:
                raise
            # Only the error class is logged: the message can contain connection details.
            logging.warning("Database not ready (attempt %d/%d): %s", attempt, attempts, type(exc).__name__)
            time.sleep(delay)
