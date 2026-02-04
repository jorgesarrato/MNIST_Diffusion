import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DATA_DIR = os.getenv("DATA_DIR")
