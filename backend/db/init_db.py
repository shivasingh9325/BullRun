from .session import engine, Base
from .models import DBPortfolioState, DBTrade, DBSystemAudit
from backend.utils.logger import get_logger

logger = get_logger(__name__)

def init_db():
    logger.info("DATABASE: Initializing Supabase Tables...")
    try:
        # Create all tables defined in models.py
        Base.metadata.create_all(bind=engine)
        logger.info("DATABASE: Initialization Successful.")
    except Exception as e:
        logger.error(f"DATABASE: Initialization Failed: {e}")
        raise

if __name__ == "__main__":
    init_db()
