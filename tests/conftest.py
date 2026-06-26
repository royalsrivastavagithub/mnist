import sys
from pathlib import Path

# Add the project root to sys.path so tests can import `src`.
# This is needed because uv doesn't install the project as a package by default.
sys.path.insert(0, str(Path(__file__).parent.parent))
