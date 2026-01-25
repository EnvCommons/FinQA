from pathlib import Path

# Determine environment path based on runtime
# Production: use /orwd_data if it exists
# Local development: use current directory
if Path("/orwd_data").exists():
    ENV_PATH = Path("/orwd_data")
else:
    ENV_PATH = Path(__file__).parent
