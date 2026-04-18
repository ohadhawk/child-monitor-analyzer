"""
Allow running the monitor package directly: ``python -m monitor``.

Delegates to the CLI entry point.
"""

from .cli import main

if __name__ == "__main__":
    main()
