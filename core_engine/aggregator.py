
class MetricTracker:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.best = float('-inf')
        self.worst = float('inf')
        self.signals = {'BUY':0,'SELL':0,'HOLD':0}
        self.correct = 0  # for future use if labeled truth available
    def update(self, pnl, label=None):
        try:
            v = float(pnl)
        except Exception:
            return
        self.count += 1
        self.total += v
        if v > self.best: self.best = v
        if v < self.worst: self.worst = v
        if label in self.signals:
            self.signals[label] += 1
    def summary(self):
        avg = self.total / self.count if self.count else 0.0
        return {"avg": round(avg,4), "best": round(self.best,4) if self.count else 0.0, "worst": round(self.worst,4) if self.count else 0.0, "trades": self.count, "signals": self.signals}
