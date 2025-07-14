#!/bin/bash

echo "Starting AI Co-Scientist Frontend..."
echo "Current directory: $(pwd)"

# Check if we're in the right directory
if [ ! -d "frontend" ]; then
    echo "Error: frontend directory not found. Please run this script from the project root."
    exit 1
fi

# Change to frontend directory
cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Start the development server
echo "Starting Vite development server..."
npm run dev 