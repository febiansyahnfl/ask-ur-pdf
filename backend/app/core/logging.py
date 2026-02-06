import logging
import logging.config
from pathlib import Path
from app.config.settings import logging_config

def setup_logging():
    """Setup logging configuration"""

    # Create logs directory if not exist 
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    try:
        # Apply logging config 
        logging.config.dictConfig(logging_config)

        # Get logger
        logger = logging.getLogger("app")
        logger.info("Logging configured successfully")

        return logger
    
    except Exception as e:
        # Fallback to basic logging if config fails
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger("app")
        logger.warning(f"Failed to configure logging from YAML: {e}")
        logger.info("Using basic logging configuration")

        return logger

# Create default logger instance
logger = setup_logging() 