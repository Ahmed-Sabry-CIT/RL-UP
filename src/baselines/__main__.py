"""Entry point so `python -m baselines` works without installation."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
