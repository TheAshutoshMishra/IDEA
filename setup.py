"""
Setup script for AadhaarInsight360
Initializes the project structure and validates environment
"""

import os
import sys
from pathlib import Path

def create_directories():
    """Create necessary directories"""
    directories = [
        'data/raw',
        'data/processed',
        'data/outputs',
        'models',
        'logs',
        'notebooks',
        'tests',
        'docs'
    ]
    
    print("Creating project directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")
    
    print("\n✅ All directories created successfully!")

def check_python_version():
    """Check if Python version is compatible"""
    print("\nChecking Python version...")
    version = sys.version_info
    
    if version.major == 3 and version.minor >= 11:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro} (Compatible)")
        return True
    else:
        print(f"  ⚠️  Python {version.major}.{version.minor}.{version.micro}")
        print(f"  Recommended: Python 3.11 or higher")
        return False

def create_env_file():
    """Create .env file template"""
    env_content = """# AadhaarInsight360 Environment Variables

# Project Configuration
PROJECT_NAME=AadhaarInsight360
ENVIRONMENT=development

# Data Paths
DATA_PATH=data/raw
OUTPUT_PATH=data/outputs

# API Keys (if needed)
# API_KEY=your_api_key_here

# Database (if needed)
# DB_HOST=localhost
# DB_PORT=5432
# DB_NAME=aadhaar_db
"""
    
    if not Path('.env').exists():
        print("\nCreating .env file...")
        with open('.env', 'w') as f:
            f.write(env_content)
        print("  ✓ .env file created")
    else:
        print("\n  ℹ️  .env file already exists")

def display_next_steps():
    """Display next steps for the user"""
    print("\n" + "="*60)
    print("🎉 AadhaarInsight360 Setup Complete!")
    print("="*60)
    
    print("\n📋 Next Steps:\n")
    
    print("1️⃣  Create Virtual Environment:")
    print("   python -m venv venv")
    print("   venv\\Scripts\\activate  # Windows")
    print("   source venv/bin/activate  # Linux/Mac")
    
    print("\n2️⃣  Install Dependencies:")
    print("   pip install -r requirements.txt")
    
    print("\n3️⃣  Place Your Data:")
    print("   - Copy UIDAI datasets to data/raw/ folder")
    print("   - Supported formats: CSV, Excel, Parquet")
    
    print("\n4️⃣  Launch Dashboard:")
    print("   streamlit run dashboard/app.py")
    
    print("\n5️⃣  Run Analysis:")
    print("   python src/main_analysis.py")
    
    print("\n📚 Documentation:")
    print("   - README.md - Full project documentation")
    print("   - QUICKSTART.md - Quick start guide")
    print("   - PRESENTATION_GUIDE.md - Hackathon presentation tips")
    print("   - PROJECT_SUMMARY.md - Project summary and functions")
    
    print("\n💡 Tips:")
    print("   - Use sample data for demo if actual data not available")
    print("   - Dashboard works without data (generates sample data)")
    print("   - Check config.yaml for customization options")
    
    print("\n" + "="*60)
    print("Good luck with your hackathon! 🚀")
    print("="*60 + "\n")

def main():
    """Main setup function"""
    print("="*60)
    print("AadhaarInsight360 - Setup Script")
    print("="*60)
    
    # Check Python version
    python_ok = check_python_version()
    
    # Create directories
    create_directories()
    
    # Create .env file
    create_env_file()
    
    # Display next steps
    display_next_steps()
    
    if not python_ok:
        print("\n⚠️  Warning: Python version may not be fully compatible")
        print("Consider upgrading to Python 3.11 or higher\n")

if __name__ == "__main__":
    main()
