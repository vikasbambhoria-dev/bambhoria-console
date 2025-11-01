"""
Minimal Flask app with Zerodha OAuth wiring so you can go live fast.
Includes:
- Login redirect to Zerodha
- Callback handler to create access_token
- Simple start_trading API (placeholder) that requires authentication
"""

import os
import json
import hashlib
import random
from datetime import datetime
from typing import Optional, Dict, Any
import threading
import time

import requests
from flask import Flask, render_template, redirect, url_for, request, jsonify, send_file
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file at the very top

def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(name)
    return val if val not in (None, "") else default


def _checksum(api_key: str, request_token: str, api_secret: str) -> str:
	raw = (api_key + request_token + api_secret).encode("utf-8")
	return hashlib.sha256(raw).hexdigest()


def create_app() -> Flask:
	app = Flask(__name__, static_folder="static", template_folder="templates")
	app.secret_key = _env("FLASK_SECRET_KEY", "bambhoria_quantum_secret_2025")

	# Zerodha config
	app.config["ZERODHA_API_KEY"] = _env("ZERODHA_API_KEY", "")
	app.config["ZERODHA_API_SECRET"] = _env("ZERODHA_API_SECRET", "")
	# Use domain callback if set, else local dev callback
	domain = _env("DOMAIN", "")
	default_callback = "http://127.0.0.1:5000/zerodha/callback"
	app.config["ZERODHA_REDIRECT_URL"] = (
		f"https://{domain}/callback" if domain else default_callback
	)

	# In-memory session
	app.config["ZERODHA_SESSION"] = None
	# Live runner state
	app.config["LIVE_RUNNER"] = {
		"running": False,
		"thread": None,
		"last_beat": None,
		"last_error": None,
		"last_quotes": None,
		"last_profile": None,
		"symbols": ["NSE:RELIANCE", "NSE:INFY"],
		"demo_mode": False,
		"demo_state": {},
		"auto_heal": True,
	}

	# In-memory recent events
	app.config["EVENTS"] = []  # list of dicts: {ts, ts_iso, level, type, message, data}

	def _log_event(level: str, etype: str, message: str, data: Optional[Dict[str, Any]] = None):
		entry = {
			"ts": int(time.time()),
			"ts_iso": datetime.utcnow().isoformat(timespec="seconds") + "Z",
			"level": level,
			"type": etype,
			"message": message,
			"data": data or {},
		}
		buf = app.config.get("EVENTS", [])
		buf.append(entry)
		if len(buf) > 200:
			app.config["EVENTS"] = buf[-200:]
		# also append to a log file (best-effort)
		try:
			with open(".events.log", "a", encoding="utf-8") as f:
				f.write(json.dumps(entry) + "\n")
		except Exception:
			pass

	# Load dashboard settings if present (symbols, demo_mode)
	try:
		if os.path.exists(".dashboard_settings.json"):
			with open(".dashboard_settings.json", "r", encoding="utf-8") as f:
				settings = json.load(f)
				run = app.config["LIVE_RUNNER"]
				if isinstance(settings.get("symbols"), list) and settings["symbols"]:
					run["symbols"] = settings["symbols"]
				run["demo_mode"] = bool(settings.get("demo_mode", False))
				run["auto_heal"] = bool(settings.get("auto_heal", True))
	except Exception:
		pass
	# Try to restore previous authenticated session (if any)
	try:
		if os.path.exists(".zerodha_session.json"):
			with open(".zerodha_session.json", "r", encoding="utf-8") as f:
				app.config["ZERODHA_SESSION"] = json.load(f)
	except Exception:
		pass

	@app.route("/")
	def index():
		return render_template("index.html")

	@app.route("/health")
	def health():
		return {"status": "ok", "app": "bambhoria-quantum", "version": "live"}

	# ---------- Zerodha OAuth ----------
	@app.route("/login/zerodha")
	def login_zerodha():
		api_key = app.config["ZERODHA_API_KEY"]
		if not api_key:
			msg = "Missing ZERODHA_API_KEY in environment. Update .env and restart."
			_log_event("error", "auth", msg)
			return (msg, 500)
		login_url = f"https://kite.trade/connect/login?api_key={api_key}&v=3"
		_log_event("info", "auth", "Redirecting to Zerodha login")
		return redirect(login_url)

	@app.route("/zerodha/callback")
	def zerodha_callback():
		try:
			_log_event("info", "auth.callback", "Received callback from Zerodha.")
			api_key = app.config["ZERODHA_API_KEY"]
			api_secret = app.config["ZERODHA_API_SECRET"]
			if not api_key or not api_secret:
				msg = "CRITICAL: API key/secret are missing in the application config."
				_log_event("error", "auth.callback", msg)
				return (msg, 500)
			_log_event("info", "auth.callback", "API credentials found in config.")

			request_token = request.args.get("request_token")
			if not request_token:
				msg = "Callback failed: request_token was not found in the URL."
				_log_event("error", "auth.callback", msg)
				return (msg, 400)
			_log_event("info", "auth.callback", "Request token received.", {"token_prefix": (request_token or "")[:8]})

			chk = _checksum(api_key, request_token, api_secret)
			_log_event("info", "auth.callback", "Generated checksum for token exchange.")
			url = "https://api.kite.trade/session/token"
			headers = {"X-Kite-Version": "3"}
			data = {
				"api_key": api_key,
				"request_token": request_token,
				"checksum": chk,
			}

			try:
				_log_event("info", "auth.callback", "Posting to Zerodha for session token...")
				resp = requests.post(url, data=data, headers=headers, timeout=15)
				_log_event("info", "auth.callback", f"Zerodha response status: {resp.status_code}")
			except Exception as e:
				msg = f"Network error during token exchange: {e}"
				_log_event("error", "auth.callback", msg)
				return (msg, 500)

			if resp.status_code != 200:
				msg = f"Zerodha token exchange failed: {resp.status_code} {resp.text}"
				_log_event("error", "auth.callback", msg)
				return (msg, 400)

			body = resp.json()
			if body.get("status") != "success" or "data" not in body:
				msg = f"Unexpected response from Zerodha: {body}"
				_log_event("error", "auth.callback", msg)
				return (msg, 400)

			session_data = body["data"]
			app.config["ZERODHA_SESSION"] = session_data

			# On successful auth, disable demo mode if it was on
			run = app.config["LIVE_RUNNER"]
			if run.get("demo_mode"):
				run["demo_mode"] = False
				_log_event("info", "auth", "Zerodha login successful, Demo Mode disabled.")

			# persist locally as well
			try:
				with open(".zerodha_session.json", "w", encoding="utf-8") as f:
					json.dump(session_data, f, indent=2)
			except Exception:
				pass

			# Auto-start live runner on successful auth
			try:
				started = _start_live_runner_if_needed()
				if started:
					_log_event("info", "runner", "Live runner started after OAuth callback")
			except Exception as e:
				_log_event("warn", "runner", f"Runner not started post-auth: {e}")

			# Redirect to a simple success page (index for now)
			return redirect(url_for("index"))
		except Exception as e:
			# THIS IS THE ULTIMATE FALLBACK TO CATCH ANY UNEXPECTED CRASH
			_log_event("critical", "auth.callback.crash", f"FATAL CRASH in callback handler: {e}", {"error_type": str(type(e))})
			# Return a generic error to the user's browser
			return "A critical server error occurred during authentication. Please check the application logs.", 500

	# Backward/compatibility aliases so external settings like
	# https://bambhoriaquantum.in/callback keep working
	@app.route("/callback")
	def callback_alias():
		# Delegate to the canonical handler
		return zerodha_callback()

	# Older docs may reference /login, keep it pointing to Zerodha login
	@app.route("/login")
	def login_alias():
		return redirect(url_for("login_zerodha"))

	@app.route("/api/status")
	def api_status():
		sess = app.config.get("ZERODHA_SESSION")
		run = app.config.get("LIVE_RUNNER", {})
		return jsonify(
			{
				"authenticated": bool(sess),
				"user_id": (sess or {}).get("user_id"),
				"has_access_token": bool((sess or {}).get("access_token")),
				"live_running": bool(run.get("running")),
				"last_beat": run.get("last_beat"),
				"last_error": run.get("last_error"),
				"symbols": run.get("symbols", []),
				"demo_mode": bool(run.get("demo_mode")),
				"auto_heal": bool(run.get("auto_heal", True)),
			}
		)

	@app.route("/api/events")
	def api_events():
		try:
			limit = int(request.args.get("limit", "50"))
		except Exception:
			limit = 50
		events = app.config.get("EVENTS", [])
		return jsonify({"ok": True, "events": events[-limit:]})

	@app.route("/api/logs/events/download")
	def api_download_events_log():
		path = ".events.log"
		if not os.path.exists(path):
			return jsonify({"ok": False, "message": "No events log yet."}), 404
		return send_file(path, as_attachment=True, download_name="events.log", mimetype="text/plain")

	def _kite_headers() -> Dict[str, str]:
		sess = app.config.get("ZERODHA_SESSION") or {}
		api_key = app.config.get("ZERODHA_API_KEY", "")
		access_token = sess.get("access_token", "")
		return {
			"X-Kite-Version": "3",
			"Authorization": f"token {api_key}:{access_token}",
		}

	def _get_profile() -> Dict[str, Any]:
		resp = requests.get(
			"https://api.kite.trade/user/profile",
			headers=_kite_headers(),
			timeout=15,
		)
		resp.raise_for_status()
		return resp.json()

	def _get_quotes(instruments: list[str]) -> Dict[str, Any]:
		params = []
		for s in instruments:
			params.append(("i", s))
		resp = requests.get(
			"https://api.kite.trade/quote",
			headers=_kite_headers(),
			params=params,
			timeout=15,
		)
		resp.raise_for_status()
		return resp.json()

	def _live_loop():
		"""Minimal live loop: verify profile then poll quotes periodically."""
		run = app.config["LIVE_RUNNER"]
		try:
			# One-time profile check
			_prof = _get_profile()
			run["last_profile"] = _prof
			run["last_error"] = None
			run["last_beat"] = int(time.time())
			_log_event("info", "runner", "Verified profile; entering live quote loop")
			# Poll quotes every 5s
			while run.get("running"):
				try:
					quotes = _get_quotes(run.get("symbols", []))
					run["last_quotes"] = quotes
					run["last_error"] = None
					run["last_beat"] = int(time.time())
				except Exception as ie:
					run["last_error"] = str(ie)
					_log_event("error", "quotes", f"Quote fetch error: {ie}")
				time.sleep(5)
		except Exception as e:
			run["last_error"] = str(e)
			_log_event("error", "runner", f"Live loop error: {e}")
		finally:
			run["running"] = False
			_log_event("warn", "runner", "Live runner stopped")

	def _demo_loop():
		"""Demo loop: generate mock quotes for current symbols without Zerodha auth."""
		run = app.config["LIVE_RUNNER"]
		state: dict = run.setdefault("demo_state", {})
		# initialize per-symbol state
		for s in run.get("symbols", []):
			if s not in state:
				price = random.uniform(100, 2500)
				state[s] = {
					"open": price,
					"high": price,
					"low": price,
					"close": price,
					"last_price": price,
				}
		try:
			while run.get("running") and run.get("demo_mode"):
				data = {}
				for s in run.get("symbols", []):
					st = state.setdefault(s, {
						"open": random.uniform(100, 2500),
						"high": 0.0,
						"low": 0.0,
						"close": 0.0,
						"last_price": 0.0,
					})
					# random walk
					delta = random.uniform(-1.5, 1.5)
					st["last_price"] = max(1.0, st["last_price"] + delta) if st["last_price"] else st["open"]
					st["high"] = max(st["high"], st["last_price"]) if st["high"] else st["last_price"]
					st["low"] = min(st["low"], st["last_price"]) if st["low"] else st["last_price"]
					st["close"] = st["last_price"]
					data[s] = {
						"last_price": round(st["last_price"], 2),
						"ohlc": {
							"open": round(st["open"], 2),
							"high": round(st["high"], 2),
							"low": round(st["low"], 2),
							"close": round(st["close"], 2),
						},
					}
				run["last_quotes"] = {"status": "success", "data": data}
				run["last_error"] = None
				run["last_beat"] = int(time.time())
				time.sleep(2)
		except Exception as e:
			run["last_error"] = str(e)
			_log_event("error", "demo", f"Demo loop error: {e}")
		finally:
			run["running"] = False
			_log_event("warn", "demo", "Demo runner stopped")

	@app.route("/api/live_quotes")
	def api_live_quotes():
		run = app.config.get("LIVE_RUNNER", {})
		return jsonify(run.get("last_quotes") or {})

	@app.route("/api/profile")
	def api_profile():
		run = app.config.get("LIVE_RUNNER", {})
		# return cached if exists
		prof = run.get("last_profile")
		if prof:
			return jsonify(prof)
		# else try fetch live if authenticated
		try:
			prof = _get_profile()
			run["last_profile"] = prof
			return jsonify(prof)
		except Exception as e:
			return jsonify({"ok": False, "error": str(e)}), 400

	@app.route("/api/set_symbols", methods=["GET", "POST"])
	def api_set_symbols():
		run = app.config.get("LIVE_RUNNER", {})
		syms: Optional[str] = None
		if request.method == "POST":
			if request.is_json:
				j = request.get_json(silent=True) or {}
				syms = j.get("symbols")
			else:
				syms = request.form.get("symbols")
		else:
			syms = request.args.get("symbols")
		if not syms:
			return jsonify({"ok": False, "message": "Provide symbols as CSV via 'symbols'"}), 400
		# normalize symbols list
		new_list = [s.strip() for s in syms.split(",") if s.strip()]
		if not new_list:
			return jsonify({"ok": False, "message": "No valid symbols provided"}), 400
		run["symbols"] = new_list
		try:
			_log_event("info", "settings", "Symbols updated", {"symbols": new_list})
		except Exception:
			pass
		# persist to settings
		try:
			with open(".dashboard_settings.json", "w", encoding="utf-8") as f:
				json.dump({
					"symbols": run.get("symbols", []),
					"demo_mode": run.get("demo_mode", False),
					"auto_heal": run.get("auto_heal", True)
				}, f, indent=2)
		except Exception:
			pass
		return jsonify({"ok": True, "symbols": new_list})

	def _start_live_runner_if_needed():
		run = app.config["LIVE_RUNNER"]
		if run.get("running"):
			return False
		# If authenticated, prefer real live loop
		sess = app.config.get("ZERODHA_SESSION")
		if sess and sess.get("access_token"):
			run["running"] = True
			thr = threading.Thread(target=_live_loop, daemon=True)
			run["thread"] = thr
			thr.start()
			_log_event("info", "runner", "Started live runner (Zerodha)")
			return True
		# Else if demo mode is enabled, start demo loop
		if run.get("demo_mode"):
			run["running"] = True
			thr = threading.Thread(target=_demo_loop, daemon=True)
			run["thread"] = thr
			thr.start()
			_log_event("info", "demo", "Started demo runner")
			return True
		raise RuntimeError("Not authenticated with Zerodha. Enable Demo Mode to simulate.")

	# Supervisor thread: keeps the runner alive if auto_heal is enabled
	def _supervisor_loop():
		run = app.config["LIVE_RUNNER"]
		while True:
			try:
				if run.get("auto_heal", True) and not run.get("running"):
					# Start if possible (auth or demo)
					try:
						_started = _start_live_runner_if_needed()
						if _started:
							run["last_error"] = None
							_log_event("info", "supervisor", "Auto-heal restarted runner")
					except Exception as ie:
						# stay quiet; expose via status if needed
						pass
			except Exception:
				pass
			time.sleep(5)

	# start supervisor as a daemon
	try:
		thr_sup = threading.Thread(target=_supervisor_loop, daemon=True)
		thr_sup.start()
	except Exception:
		pass

	@app.route("/api/start_trading", methods=["GET", "POST"]) 
	def api_start_trading():
		try:
			started = _start_live_runner_if_needed()
			if started:
				_log_event("info", "runner", "Start requested -> started")
				return jsonify({"ok": True, "message": "Live trading loop started."})
			_log_event("info", "runner", "Start requested -> already running")
			return jsonify({"ok": True, "message": "Live already running."})
		except Exception as e:
			_log_event("warn", "runner", f"Start requested -> blocked: {e}")
			return jsonify({"ok": False, "message": str(e)}), 401

	@app.route("/api/stop_trading", methods=["GET", "POST"])
	def api_stop_trading():
		run = app.config.get("LIVE_RUNNER", {})
		run["running"] = False
		_log_event("info", "runner", "Stop requested")
		return jsonify({"ok": True, "message": "Live trading loop stop requested."})

	@app.route("/api/toggle_demo", methods=["POST"]) 
	def api_toggle_demo():
		run = app.config.get("LIVE_RUNNER", {})
		try:
			payload = request.get_json(silent=True) or {}
			enable = bool(payload.get("enable"))
			run["demo_mode"] = enable
			_log_event("info", "settings", f"Demo mode set to {enable}")
			# persist to settings
			try:
				with open(".dashboard_settings.json", "w", encoding="utf-8") as f:
					json.dump({
						"symbols": run.get("symbols", []),
						"demo_mode": run.get("demo_mode", False),
						"auto_heal": run.get("auto_heal", True)
					}, f, indent=2)
			except Exception:
				pass
			return jsonify({"ok": True, "demo_mode": run["demo_mode"]})
		except Exception as e:
			return jsonify({"ok": False, "message": str(e)}), 400

	@app.route("/api/toggle_autoheal", methods=["POST"]) 
	def api_toggle_autoheal():
		run = app.config.get("LIVE_RUNNER", {})
		try:
			payload = request.get_json(silent=True) or {}
			enable = bool(payload.get("enable"))
			run["auto_heal"] = enable
			_log_event("info", "settings", f"Auto-heal set to {enable}")
			# persist to settings
			try:
				with open(".dashboard_settings.json", "w", encoding="utf-8") as f:
					json.dump({
						"symbols": run.get("symbols", []),
						"demo_mode": run.get("demo_mode", False),
						"auto_heal": run.get("auto_heal", True)
					}, f, indent=2)
			except Exception:
				pass
			return jsonify({"ok": True, "auto_heal": run["auto_heal"]})
		except Exception as e:
			return jsonify({"ok": False, "message": str(e)}), 400

	@app.route("/api/live_status")
	def api_live_status():
		run = app.config.get("LIVE_RUNNER", {})
		return jsonify(
			{
				"running": bool(run.get("running")),
				"last_beat": run.get("last_beat"),
				"last_error": run.get("last_error"),
				"has_quotes": bool(run.get("last_quotes")),
			}
		)

	return app


# Flask app instance for WSGI
app = create_app()

if __name__ == "__main__":
	app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)

