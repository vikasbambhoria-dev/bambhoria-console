"""
🔥 MANUAL API CREDENTIALS UPDATE 🔥
===================================

यहाँ अपनी Zerodha API credentials दालें:
"""

# 🔑 Your Zerodha API Credentials
API_KEY = "YOUR_API_KEY_HERE"           # आपकी API Key यहाँ दालें
API_SECRET = "YOUR_API_SECRET_HERE"     # आपकी API Secret यहाँ दालें

import os
from pathlib import Path

def update_env_file():
    """Update .env file with API credentials"""
    
    if API_KEY == "YOUR_API_KEY_HERE" or API_SECRET == "YOUR_API_SECRET_HERE":
        print("❌ Please update API_KEY and API_SECRET above first!")
        return False
    
    env_file = Path(".env")
    
    try:
        # Read current .env file
        if env_file.exists():
            with open(env_file, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            content = ""
        
        # Update credentials
        lines = content.split('\n')
        updated_lines = []
        api_key_found = False
        api_secret_found = False
        
        for line in lines:
            if line.startswith('ZERODHA_API_KEY='):
                updated_lines.append(f'ZERODHA_API_KEY={API_KEY}')
                api_key_found = True
            elif line.startswith('ZERODHA_API_SECRET='):
                updated_lines.append(f'ZERODHA_API_SECRET={API_SECRET}')
                api_secret_found = True
            else:
                updated_lines.append(line)
        
        # Add if not found
        if not api_key_found:
            updated_lines.append(f'ZERODHA_API_KEY={API_KEY}')
        if not api_secret_found:
            updated_lines.append(f'ZERODHA_API_SECRET={API_SECRET}')
        
        # Write back
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(updated_lines))
        
        print("✅ API Credentials successfully updated in .env file!")
        print(f"   📁 File: {env_file.absolute()}")
        print(f"   🔑 API Key: {API_KEY}")
        print(f"   🔒 API Secret: {'*' * len(API_SECRET)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error updating .env file: {e}")
        return False

if __name__ == "__main__":
    print("🔥 BAMBHORIA QUANTUM API CREDENTIALS UPDATE 🔥")
    print("="*50)
    
    print("\n📋 Instructions:")
    print("1. Edit this file (manual_api_update.py)")
    print("2. Replace API_KEY and API_SECRET values above")
    print("3. Save the file")
    print("4. Run this script again")
    
    print(f"\n🔍 Current values:")
    print(f"   API_KEY = {API_KEY}")
    print(f"   API_SECRET = {API_SECRET}")
    
    if update_env_file():
        print("\n🎉 SUCCESS! Your Bambhoria Quantum platform is ready!")
        print("\n🚀 Next steps:")
        print("1. Run: python bambhoria_quantum_web_app.py")
        print("2. Access: https://bambhoriaquantum.in")
        print("3. Connect with Zerodha and start trading!")
    else:
        print("\n⚠️ Please update the credentials in this file first.")