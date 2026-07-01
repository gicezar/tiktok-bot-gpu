import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from bot.telegram_bot import run_bot

if __name__ == "__main__":
    run_bot()
