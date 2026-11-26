"""CLI wrapper for Drift Monitor Worker."""
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from services.drift_monitor.worker import DriftMonitorWorker, main

if __name__ == "__main__":
    main()
