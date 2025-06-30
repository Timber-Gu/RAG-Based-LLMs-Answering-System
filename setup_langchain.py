"""
Setup script for LangChain ML Q&A Assistant
Simplified setup focusing on LangChain multi-agent system
"""
import os
import sys
import subprocess
from pathlib import Path

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = Path('.env')
    env_example = Path('env.example')
    
    if not env_file.exists():
        if env_example.exists():
            # Copy from example
            env_file.write_text(env_example.read_text())
            print("✅ Created .env file from env.example")
        else:
            # Create basic .env file
            env_content = """# OpenAI API Key (required)
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Uncomment to customize
# OPENAI_MODEL=gpt-3.5-turbo
# AGENT_TEMPERATURE=0.7
"""
            env_file.write_text(env_content)
            print("✅ Created .env file template")
        
        print("⚠️ Please edit .env file with your OpenAI API key")
        return False
    else:
        print("✅ .env file already exists")
        return True

def create_directories():
    """Create necessary directories"""
    directories = [
        'data',
        'data/papers',
        'data/vector_store'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Directory created/verified: {directory}")

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True, capture_output=True, text=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_api_key():
    """Check if OpenAI API key is configured"""
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key == 'your_openai_api_key_here':
        print("⚠️ OpenAI API key not configured")
        print("Please edit .env file with your actual API key")
        return False
    else:
        print("✅ OpenAI API key configured")
        return True

def test_langchain_agents():
    """Test LangChain agents initialization"""
    try:
        # Add src to path
        sys.path.append('src')
        from src.agents.langchain_agents import create_langchain_ml_agents
        
        print("🧪 Testing LangChain agents...")
        agents = create_langchain_ml_agents()
        
        # Test health check
        health = agents.health_check()
        print(f"🔍 Health check results: {health}")
        
        if health.get('llm_connection'):
            print("✅ LangChain agents working correctly!")
            return True
        else:
            print("⚠️ LangChain agents initialized but LLM connection failed")
            return False
    
    except Exception as e:
        print(f"❌ Error testing LangChain agents: {e}")
        return False

def main():
    """Main setup function"""
    print("🌟 LangChain ML Q&A Assistant Setup")
    print("=" * 50)
    
    # Step 1: Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Step 2: Install dependencies
    print("\n📦 Installing dependencies...")
    if not install_dependencies():
        print("❌ Setup failed at dependency installation")
        return
    
    # Step 3: Create .env file
    print("\n🔧 Setting up environment...")
    env_exists = create_env_file()
    
    if not env_exists:
        print("\n⚠️ Please configure your .env file and run setup again")
        return
    
    # Step 4: Check API key
    print("\n🔑 Checking API configuration...")
    if not check_api_key():
        print("\n⚠️ Please configure your OpenAI API key in .env file")
        return
    
    # Step 5: Test LangChain agents
    print("\n🧪 Testing system...")
    if test_langchain_agents():
        print("\n🎉 Setup completed successfully!")
        print("\n🚀 You can now:")
        print("  • Run 'python main.py' for interactive mode")
        print("  • Run 'python -m src.api.langchain_server' for API server")
        print("  • Access API docs at http://localhost:8000/docs")
    else:
        print("\n⚠️ Setup completed with warnings")
        print("Please check your OpenAI API key and internet connection")

if __name__ == "__main__":
    main() 