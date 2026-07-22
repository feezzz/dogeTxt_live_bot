"""Simple launcher for the live signal bot."""
import sys
import os

# This repo is designed to sit alongside event_backtest/ in the same parent directory.
# Clone as: git clone ... dogeTxt_live_bot
# And ensure event_backtest/ is also present in the parent directory.
PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ_ROOT)

from main import main  # noqa: E402
main()
