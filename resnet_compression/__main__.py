"""Allows ``python -m resnet_compression`` as an alias for the ``resnet-compress`` CLI."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
