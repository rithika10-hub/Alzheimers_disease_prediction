import logging
import os
from datetime import datetime

# Define log filename
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

# Define log directory
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Full log file path
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Configure logging (Make sure no duplicate 'level' argument)
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,  # Ensure this is only included once
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",  # Overwrites log file every time
)

# Define logger
logger = logging.getLogger()
