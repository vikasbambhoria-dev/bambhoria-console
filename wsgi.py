#!/usr/bin/env python3
# Bambhoria Quantum WSGI Application

import os
import sys
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import Flask application
from bambhoria_quantum_web_app import app

# WSGI application
application = app

if __name__ == "__main__":
    # For local development
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    )
