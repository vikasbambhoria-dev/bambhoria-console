
from typing import Sequence, Optional
import random

class HeuristicModel:
    def __init__(self, seed: Optional[int]=None):
        if seed is not None:
            random.seed(seed)
    def features_to_score(self, features: Sequence[float]) -> float:
        if not features:
            return 0.5
        price = float(features[0]) if len(features)>0 and features[0] is not None else 0.0
        sma5 = float(features[1]) if len(features)>1 and features[1] is not None else price
        sma20 = float(features[2]) if len(features)>2 and features[2] is not None else price
        ema12 = float(features[3]) if len(features)>3 and features[3] is not None else price
        ema26 = float(features[4]) if len(features)>4 and features[4] is not None else price
        rsi14 = float(features[5]) if len(features)>5 and features[5] is not None else 50.0
        momentum = float(features[6]) if len(features)>6 and features[6] is not None else 0.0
        mom_signal = 1.0 if momentum > 0 else 0.0 if momentum < 0 else 0.5
        trend_score = 0.5
        try:
            trend_score = 0.5 + 0.25 * (1 if sma5 > sma20 else -1)
            trend_score += 0.25 * (1 if ema12 and ema26 and ema12 > ema26 else -1)
            trend_score = max(0.0, min(1.0, trend_score))
        except Exception:
            trend_score = 0.5
        rsi_adj = 0.5
        try:
            if rsi14 is not None:
                if rsi14 < 30:
                    rsi_adj = 0.75
                elif rsi14 > 70:
                    rsi_adj = 0.25
                else:
                    rsi_adj = 0.5 + (rsi14 - 50) / 100.0
        except Exception:
            rsi_adj = 0.5
        score = 0.5 * trend_score + 0.3 * rsi_adj + 0.2 * mom_signal
        score = max(0.0, min(1.0, score + random.uniform(-0.02, 0.02)))
        return score
    def predict_proba(self, features: Sequence[float]):
        score = self.features_to_score(features)
        return [[1-score, score]]
    def predict_label(self, features: Sequence[float]):
        p = self.predict_proba(features)[0][1]
        if p > 0.6:
            return "BUY"
        if p < 0.4:
            return "SELL"
        return "HOLD"
class ModelWrapper:
    def __init__(self, model):
        self.model = model
    def predict_proba(self, X):
        try:
            return self.model.predict_proba(X)
        except Exception:
            preds = self.model.predict(X)
            return [[1 - float(p), float(p)] for p in preds]
    def predict_label(self, X):
        proba = self.predict_proba(X)
        p = proba[0][1] if proba and len(proba) and len(proba[0])>1 else 0.5
        if p > 0.6: return "BUY"
        if p < 0.4: return "SELL"
        return "HOLD"
def create_model_from_loader(loaded_model):
    if loaded_model is None:
        return HeuristicModel()
    if hasattr(loaded_model, 'predict'):
        return ModelWrapper(loaded_model)
    return HeuristicModel()
