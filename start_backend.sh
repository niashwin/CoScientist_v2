#!/bin/bash

echo "Starting AI Co-Scientist Backend..."
echo "Current directory: $(pwd)"

# Check if we're in the right directory
if [ ! -d "backend" ]; then
    echo "Error: backend directory not found. Please run this script from the project root."
    exit 1
fi

# Set PYTHONPATH to include the project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Start the backend server from the project root
echo "Starting FastAPI server..."
cd backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000 