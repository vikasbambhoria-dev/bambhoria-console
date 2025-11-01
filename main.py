
#!/usr/bin/env python3
import os, time, argparse, json
from core_engine import model_loader as model_loader_module, ai_quant_model as ai_quant_module
from core_engine.aggregator import MetricTracker
def info(msg): print("[INFO]", msg)
def _get_model_score(model_obj, features):
    try:
        proba = model_obj.predict_proba([features])
        if isinstance(proba, (list,tuple)) and proba and isinstance(proba[0], (list,tuple)):
            p = float(proba[0][1])
            label = model_obj.predict_label(features) if hasattr(model_obj,'predict_label') else ("BUY" if p>0.5 else "SELL")
            return p, label
    except Exception:
        pass
    try:
        proba = model_obj.predict_proba(features)
        if isinstance(proba, (list,tuple)) and proba and isinstance(proba[0], (list,tuple)):
            p = float(proba[0][1])
            label = model_obj.predict_label(features) if hasattr(model_obj,'predict_label') else ("BUY" if p>0.5 else "SELL")
            return p, label
    except Exception:
        pass
    try:
        preds = model_obj.predict([features])
        p = float(preds[0]) if isinstance(preds, (list,tuple)) else float(preds)
        label = model_obj.predict_label(features) if hasattr(model_obj,'predict_label') else ("BUY" if p>0.5 else "SELL")
        return p, label
    except Exception:
        return 0.5, "HOLD"
def run_live_mode():
    info("Starting LIVE mode...")
    ws_url = os.environ.get("GODEYE_WS_URL") or os.environ.get("GOD_EYE_WS_URL")
    model_path = os.environ.get("GODEYE_MODEL_PATH")
    ml_loader = model_loader_module.ModelLoader(model_path)
    loaded_model = ml_loader.model
    model_obj = ai_quant_module.create_model_from_loader(loaded_model)
    info(f"[AI] Model loaded: {'real' if loaded_model is not None else 'heuristic placeholder'} (path={model_path})")
    tracker = MetricTracker()
    # connect to source and forward augmented payloads to local dashboard forwarder endpoint if set
    import asyncio, websockets
    async def client_loop():
        async with websockets.connect(ws_url) as ws:
            async for msg in ws:
                try:
                    data = json.loads(msg)
                except Exception:
                    continue
                features = [data.get('price') or (data.get('ohlc') or {}).get('close'), data.get('sma5'), data.get('sma20'), data.get('ema12'), data.get('ema26'), data.get('rsi14'), data.get('momentum'), data.get('volume')]
                score, label = _get_model_score(model_obj, features)
                live_pnl = data.get('live_pnl', round((score-0.5)*100,2))
                tracker.update(live_pnl, label)
                data['score'] = float(round(score,4))
                data['label'] = label
                data['pnl'] = float(round(live_pnl,4))
                data['agg'] = tracker.summary()
                print(json.dumps(data))
    asyncio.get_event_loop().run_until_complete(client_loop())

def run_mock_mode(duration_seconds: int = 60, interval: float = 0.5):
    info("Starting MOCK mode (no WebSocket source detected)...")
    # Model init same as live
    ws_url = os.environ.get("GODEYE_WS_URL") or os.environ.get("GOD_EYE_WS_URL")
    model_path = os.environ.get("GODEYE_MODEL_PATH")
    ml_loader = model_loader_module.ModelLoader(model_path)
    loaded_model = ml_loader.model
    model_obj = ai_quant_module.create_model_from_loader(loaded_model)
    info(f"[AI] Model loaded: {'real' if loaded_model is not None else 'heuristic placeholder'} (path={model_path})")
    tracker = MetricTracker()

    # Simple synthetic generator
    import math, random
    start = time.time()
    t = 0
    price = 100.0
    sma5 = sma20 = ema12 = ema26 = price
    rsi = 50.0
    momentum = 0.0
    volume = 100

    def ema(prev, val, k):
        return prev * (1 - k) + val * k

    while duration_seconds <= 0 or (time.time() - start) < duration_seconds:
        # generate walk
        drift = math.sin(t / 20.0) * 0.2
        shock = random.uniform(-0.3, 0.3)
        price = max(1.0, price * (1 + (drift + shock) / 100.0))
        sma5 = (sma5 * 4 + price) / 5.0
        sma20 = (sma20 * 19 + price) / 20.0
        ema12 = ema(ema12, price, 2 / (12 + 1))
        ema26 = ema(ema26, price, 2 / (26 + 1))
        rsi = 50.0 + math.tanh((price - sma20) / max(1.0, sma20) * 10) * 20
        momentum = price - sma20
        volume = max(1, int(volume * (1 + random.uniform(-0.1, 0.1))))

        features = [price, sma5, sma20, ema12, ema26, rsi, momentum, volume]
        score, label = _get_model_score(model_obj, features)
        live_pnl = round((score - 0.5) * 100, 2)
        tracker.update(live_pnl, label)
        payload = {
            "price": round(price, 2),
            "sma5": round(sma5, 2),
            "sma20": round(sma20, 2),
            "ema12": round(ema12, 2),
            "ema26": round(ema26, 2),
            "rsi14": round(rsi, 2),
            "momentum": round(momentum, 4),
            "volume": volume,
            "score": float(round(score, 4)),
            "label": label,
            "pnl": float(round(live_pnl, 4)),
            "agg": tracker.summary(),
        }
        print(json.dumps(payload))
        t += 1
        time.sleep(interval)
if __name__ == '__main__':
    # auto-detect mode: prefer live if ws reachable
    ws = os.environ.get("GODEYE_WS_URL","ws://127.0.0.1:8765/ws")
    try:
        import socket, urllib.parse
        u = urllib.parse.urlparse(ws)
        host = u.hostname or "127.0.0.1"; port = u.port or 8765
        s = socket.socket(); s.settimeout(0.6); s.connect((host,port)); s.close()
        run_live_mode()
    except Exception:
        # Fallback to built-in mock mode for a smooth run
        run_mock_mode(duration_seconds=0)  # 0 = run indefinitely
