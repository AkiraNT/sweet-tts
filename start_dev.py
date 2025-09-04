#!/usr/bin/env python3
"""
Simple development server with auto-reload for VietTTS
"""
import os
import sys
import subprocess

def main():
    print("🚀 Starting VietTTS Development Server...")
    
    # Install package in editable mode first
    try:
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-e', '.', '--no-deps'
        ], check=True)
        print("✅ Package installed in editable mode")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Warning: Could not install package: {e}")
    
    # Setup environment
    env = os.environ.copy()
    env.update({
        'PYTHONUNBUFFERED': '1',
        'CUDA_VISIBLE_DEVICES': '',
        'FORCE_CPU': '1',
        'PYTHONPATH': '/app'
    })
    
    print("✨ Starting server with auto-reload...")
    print("   - Server: http://localhost:8298")
    print("   - Press Ctrl+C to stop")
    
    # Start server với uvicorn reload
    try:
        subprocess.run([
            'uvicorn', 
            'viettts.server:app',
            '--host', '0.0.0.0',
            '--port', '8298',
            '--reload',
            '--reload-dir', '/app/viettts',
            '--reload-dir', '/app/samples', 
            '--reload-dir', '/app/web'
        ], env=env, check=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except subprocess.CalledProcessError:
        # Fallback to original method if uvicorn reload fails
        print("⚠️  Uvicorn reload failed, using standard server...")
        subprocess.run([
            sys.executable, '-m', 'viettts.server',
            '--host', '0.0.0.0',
            '--port', '8298'
        ], env=env)

if __name__ == "__main__":
    main()