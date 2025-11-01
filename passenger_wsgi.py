"""
Passenger entrypoint for cPanel/Apache deployments.
This imports the Flask app from wsgi.py as `application`.
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Delegate to existing WSGI app
from wsgi import application  # noqa: F401
