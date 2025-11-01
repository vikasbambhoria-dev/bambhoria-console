"""
Monitor Directory Launcher
Launches the auto-heal engine from the monitor directory
"""

import subprocess
import os
import sys

def launch_auto_heal():
    """Launch auto-heal engine from monitor directory"""
    
    print("🎯 Bambhoria Auto-Heal Engine Launcher")
    print("=" * 50)
    
    # Change to monitor directory
    monitor_dir = os.path.join(os.getcwd(), "monitor")
    
    if not os.path.exists(monitor_dir):
        print("❌ Monitor directory not found!")
        return False
    
    print(f"📁 Monitor directory: {monitor_dir}")
    print("🚀 Starting Auto-Heal Engine...")
    
    try:
        # Start auto-heal engine from monitor directory
        process = subprocess.Popen(
            [sys.executable, "auto_heal_engine.py"],
            cwd=monitor_dir,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        
        print("✅ Auto-Heal Engine started successfully!")
        print(f"📊 Process ID: {process.pid}")
        print("💡 Check the new console window for monitoring logs")
        print("🛑 To stop: Close the auto-heal console window or press Ctrl+C there")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to start Auto-Heal Engine: {e}")
        return False

if __name__ == "__main__":
    success = launch_auto_heal()
    if success:
        print("\n🎉 Auto-Heal Engine is now monitoring your system!")
    else:
        print("\n❌ Failed to start monitoring system")