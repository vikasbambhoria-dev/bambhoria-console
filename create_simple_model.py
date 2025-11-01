"""
Simple Mock Model Creator
Creates a basic sklearn model that can be pickled properly
"""

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate simple training data
X, y = make_classification(n_samples=1000, n_features=3, n_classes=3, 
                          n_informative=3, n_redundant=0, random_state=42)

# Create and train a simple RF model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# Save the model
joblib.dump(model, 'models/godeye_lgbm_model.pkl')
print("✅ Compatible RandomForest model saved to models/godeye_lgbm_model.pkl")
print(f"Model classes: {model.classes_}")
print(f"Model features: {model.n_features_in_}")