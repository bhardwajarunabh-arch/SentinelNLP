"""Deprecated compatibility entry point for translation preparation.

No network translation service is called. Use this command to create the
current-test manual/external import template instead.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_translation_import import main


if __name__ == "__main__":
    main()
