"""
test_risk_manager.py
Quick test of the risk manager functionality
"""

from core_engine.risk_manager import RiskManager

def test_risk_manager():
    print("🧪 Testing Risk Manager")
    print("=" * 30)
    
    # Create risk manager
    risk_mgr = RiskManager()
    
    # Test the check_risk method
    result = risk_mgr.check_risk("TCS", 10, 3200.0, "BUY")
    print(f"Risk Check Result: {result}")
    
    # Test recording a trade
    risk_mgr.record_trade("TCS", 10, 3200.0, "BUY", 150.0)
    
    # Check risk status
    status = risk_mgr.get_risk_status()
    print(f"Risk Status: {status}")

if __name__ == "__main__":
    test_risk_manager()
