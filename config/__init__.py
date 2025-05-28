"""
Global configuration bootstrap.
Executed as soon as *anything* under `config` is imported.
"""

from pathlib import Path

from dotenv import load_dotenv, find_dotenv

# 1. Look for a .env file (repo root by default)
#    fall back to .env.docker if someone forgets to copy it.
env_file = find_dotenv(usecwd=True) or Path(__file__).parents[1] / ".env.docker"

# 2. Load variables but do NOT overwrite ones that are
#    already set by the shell / CI.
load_dotenv(dotenv_path=env_file, override=False)
