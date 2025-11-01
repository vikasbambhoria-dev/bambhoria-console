"""
Mock LightGBM Model Creator
Creates a simple mock model for testing the signal generator
"""

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Create a simple mock classifier that mimics LightGBM behavior
class MockLGBMModel:
    def __init__(self):
        # Simple rule-based model for demo
        self.feature_names = ["price_change", "volume_change", "volatility"]
    
    def predict_proba(self, X):
        """
        Mock prediction logic:
        - If price_change > 0.01 (1%): Higher BUY probability
        - If price_change < -0.01 (-1%): Higher SELL probability
        - Otherwise: HOLD
        """
        results = []
        for _, row in X.iterrows():
            price_chg = row['price_change']
            vol_chg = row['volume_change']
            volatility = row['volatility']
            
            if price_chg > 0.01 and volatility < 0.05:  # Strong up movement, low volatility
                probs = [0.2, 0.1, 0.7]  # SELL, HOLD, BUY
            elif price_chg < -0.01 and volatility < 0.05:  # Strong down movement, low volatility
                probs = [0.7, 0.1, 0.2]  # SELL, HOLD, BUY
            elif volatility > 0.05:  # High volatility - be cautious
                probs = [0.3, 0.5, 0.2]  # SELL, HOLD, BUY
            else:  # Neutral - mostly hold
                probs = [0.2, 0.6, 0.2]  # SELL, HOLD, BUY
            
            results.append(probs)
        
        return np.array(results)
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

# Create and save the mock model
mock_model = MockLGBMModel()
joblib.dump(mock_model, 'models/godeye_lgbm_model.pkl')
print("✅ Mock LightGBM model created and saved to models/godeye_lgbm_model.pkl")