"""Keep test runs away from the real saved camera list."""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("CPG44_CAMERAS_FILE", str(Path(tempfile.mkdtemp(prefix="cpg44-test-")) / "cameras.json"))
