"""
Launch script for the Streamlit ML Q&A Assistant
Provides easy access to run the web interface
"""
import os
import sys
import subprocess
from pathlib import Path

def main():
    """Main function to launch Streamlit app"""
    
    # Get the current directory (project root)
    project_root = Path(__file__).parent
    streamlit_app = project_root / "streamlit_app.py"
    
    # Check if streamlit_app.py exists
    if not streamlit_app.exists():
        print("❌ streamlit_app.py not found in the current directory")
        return
    
    # Check if .env file exists
    env_file = project_root / ".env"
    if not env_file.exists():
        print("⚠️ Warning: .env file not found")
        print("Please create a .env file with your API keys:")
        print("OPENAI_API_KEY=your_openai_key_here")
        print("ANTHROPIC_API_KEY=your_anthropic_key_here (optional)")
        print("PINECONE_API_KEY=your_pinecone_key_here (optional)")
        print("")
    
    print("🚀 Starting LangChain ML Q&A Assistant Streamlit App...")
    print(f"📁 Project directory: {project_root}")
    print("🌐 The app will open in your default web browser")
    print("⏹️ Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(streamlit_app),
            "--server.address", "localhost",
            "--server.port", "8501",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ], cwd=project_root)
        
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped by user")
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install it with: pip install streamlit")
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")

if __name__ == "__main__":
    main() 