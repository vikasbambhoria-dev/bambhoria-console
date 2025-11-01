
import os, joblib
class ModelLoader:
    def __init__(self, path=None):
        self.path = path
        self.model = None
        if path and os.path.exists(path):
            try:
                self.model = joblib.load(path)
            except Exception as e:
                print(f"[ModelLoader] failed to load {path}: {e}")
                self.model = None
