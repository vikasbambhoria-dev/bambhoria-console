"""
🔥 BAMBHORIA QUANTUM API CREDENTIALS SETUP 🔥
===============================================
Interactive setup for Zerodha API credentials
Domain: bambhoriaquantum.in
===============================================
"""

import os
import getpass
from pathlib import Path

def setup_zerodha_credentials():
    """Interactive setup for Zerodha API credentials"""
    
    print("\n" + "="*60)
    print("🔥 BAMBHORIA QUANTUM ZERODHA API SETUP 🔥")
    print("🌐 Domain: bambhoriaquantum.in")
    print("="*60 + "\n")
    
    print("📋 Zerodha API Credentials को यहाँ enter करें:")
    print("   (Get these from: https://developers.kite.trade/)\n")
    
    # Get API Key
    api_key = input("🔑 Enter your Zerodha API Key: ").strip()
    if not api_key:
        print("❌ API Key is required!")
        return False
    
    # Get API Secret
    print("\n🔒 Enter your Zerodha API Secret:")
    api_secret = getpass.getpass("   (Hidden for security): ").strip()
    if not api_secret:
        print("❌ API Secret is required!")
        return False
    
    # Confirm credentials
    print(f"\n✅ API Key: {api_key}")
    print(f"✅ API Secret: {'*' * len(api_secret)}")
    
    confirm = input("\n🎯 Are these credentials correct? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Setup cancelled. Please run again with correct credentials.")
        return False
    
    # Update .env file
    env_file_path = Path(".env")
    
    try:
        # Read current .env file
        if env_file_path.exists():
            with open(env_file_path, 'r', encoding='utf-8') as f:
                env_content = f.read()
        else:
            env_content = ""
        
        # Update API credentials
        lines = env_content.split('\n')
        updated_lines = []
        api_key_updated = False
        api_secret_updated = False
        
        for line in lines:
            if line.startswith('ZERODHA_API_KEY='):
                updated_lines.append(f'ZERODHA_API_KEY={api_key}')
                api_key_updated = True
            elif line.startswith('ZERODHA_API_SECRET='):
                updated_lines.append(f'ZERODHA_API_SECRET={api_secret}')
                api_secret_updated = True
            else:
                updated_lines.append(line)
        
        # Add credentials if not found
        if not api_key_updated:
            updated_lines.append(f'ZERODHA_API_KEY={api_key}')
        if not api_secret_updated:
            updated_lines.append(f'ZERODHA_API_SECRET={api_secret}')
        
        # Write updated .env file
        with open(env_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(updated_lines))
        
        print("\n" + "="*60)
        print("🎉 CREDENTIALS SUCCESSFULLY UPDATED! 🎉")
        print("="*60)
        
        print("\n✅ Updated files:")
        print(f"   📁 {env_file_path.absolute()}")
        
        print("\n🚀 Next steps:")
        print("1. 🌐 Go to https://developers.kite.trade/")
        print("2. 🔗 Set Redirect URL: https://bambhoriaquantum.in/callback")
        print("3. 📡 Set Postback URL: https://bambhoriaquantum.in/api/webhook/zerodha")
        print("4. ▶️ Run: python bambhoria_quantum_web_app.py")
        print("5. 🎯 Access: https://bambhoriaquantum.in")
        
        print("\n🔥 Your Bambhoria Quantum platform is ready for live trading! 🔥")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error updating credentials: {e}")
        return False

def display_current_config():
    """Display current configuration"""
    
    print("\n📋 CURRENT CONFIGURATION:")
    print("="*40)
    
    env_file_path = Path(".env")
    if env_file_path.exists():
        try:
            with open(env_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if 'API_SECRET' in line:
                            key, value = line.split('=', 1)
                            print(f"   {key}={'*' * len(value)}")
                        else:
                            print(f"   {line}")
        except Exception as e:
            print(f"   ❌ Error reading config: {e}")
    else:
        print("   ⚠️ No .env file found")

def create_env_template():
    """Create environment template if .env doesn't exist"""
    
    env_file_path = Path(".env")
    if not env_file_path.exists():
        template = """# Bambhoria Quantum Environment Configuration
# Zerodha API Credentials
ZERODHA_API_KEY=your_zerodha_api_key_here
ZERODHA_API_SECRET=your_zerodha_api_secret_here

# Flask Configuration
FLASK_SECRET_KEY=bambhoria_quantum_secret_2025_ultra_secure
FLASK_DEBUG=False
PORT=5000

# Domain Configuration
DOMAIN=bambhoriaquantum.in
SSL_ENABLED=True

# Database Configuration (if needed)
DATABASE_URL=sqlite:///bambhoria_quantum.db

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=bambhoria_quantum.log

# Trading Configuration
MAX_POSITION_SIZE=100000
MAX_DAILY_LOSS=10000
STOP_LOSS_PERCENTAGE=2.0
TAKE_PROFIT_PERCENTAGE=5.0

# Security Configuration
SESSION_TIMEOUT=3600
RATE_LIMIT_PER_MINUTE=60
"""
        with open(env_file_path, 'w', encoding='utf-8') as f:
            f.write(template)
        print(f"✅ Created .env template: {env_file_path.absolute()}")

if __name__ == "__main__":
    print("🔥 BAMBHORIA QUANTUM SETUP SCRIPT 🔥")
    
    # Create .env template if needed
    create_env_template()
    
    # Display current configuration
    display_current_config()
    
    # Setup credentials
    success = setup_zerodha_credentials()
    
    if success:
        print("\n🏆 SETUP COMPLETED SUCCESSFULLY!")
        print("🚀 Ready to launch Bambhoria Quantum trading platform!")
    else:
        print("\n⚠️ Setup incomplete. Please run again.")
    
    input("\nPress Enter to continue...")