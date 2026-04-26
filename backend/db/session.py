from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

# Load environment variables from .env
# This is assuming the .env is in the parent directory of 'app' (the backend root)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

DATABASE_URL = os.getenv("SUPABASE_DB_URL")

# Fallback/Default for development using SQLite
_USE_SQLITE = not DATABASE_URL
if _USE_SQLITE:
    DATABASE_URL = "sqlite:///./data/bullrun_api.sqlite"

# Engine args differ for SQLite vs Postgres
if _USE_SQLITE:
    # SQLite does not support connection pooling params
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False}
    )
else:
    # Optimized for Supabase (Serverless/Postgres)
    engine = create_engine(
        DATABASE_URL,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
