#!/usr/bin/env python3
"""
Simple server runner for the AI Co-Scientist backend.
Run this from the project root directory.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now we can import and run the server
if __name__ == "__main__":
    import uvicorn
    
    # Change to backend directory for relative imports
    os.chdir(project_root / "backend")
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(project_root / "backend")]
    ) 