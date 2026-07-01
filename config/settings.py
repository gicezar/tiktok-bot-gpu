import os
from dotenv import load_dotenv
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_ALLOWED_USER_ID = int(os.getenv("TELEGRAM_ALLOWED_USER_ID", "0"))
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")
AVATAR_BASE_IMAGE = os.getenv("AVATAR_BASE_IMAGE", "./assets/avatar_base.jpg")
os.makedirs(OUTPUT_DIR, exist_ok=True)
