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
from typing import Optional, Dict, Any, List
import threading
import time
import sys
import platform
import uuid
import functools
import secrets
import sqlite3

import requests
from flask import Flask, render_template, redirect, url_for, request, jsonify, send_file, Response, g, stream_with_context
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file at the very top

def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(name)
    return val if val not in (None, "") else default


def _env_bool(name: str, default: bool = False) -> bool:
	val = os.getenv(name)
	if val is None or val == "":
		return default
	return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


def _checksum(api_key: str, request_token: str, api_secret: str) -> str:
	raw = (api_key + request_token + api_secret).encode("utf-8")
	return hashlib.sha256(raw).hexdigest()

# Optional Prometheus metrics
try:
	from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
	METRICS_ENABLED = True
except Exception:
	Counter = None  # type: ignore
	Gauge = None  # type: ignore
	Histogram = None  # type: ignore
	generate_latest = None  # type: ignore
	CONTENT_TYPE_LATEST = "text/plain"  # type: ignore
	METRICS_ENABLED = False

# Define metrics placeholders (will be assigned in create_app so they have app context)
_metrics = {
	"events_total": None,
	"live_running": None,
	"demo_mode": None,
	"auto_heal": None,
	"authenticated": None,
	"last_beat": None,
	"quotes_success_total": None,
	"quotes_error_total": None,
	"profile_success_total": None,
	"profile_error_total": None,
	"rate_limited_total": None,
	"panic_total": None,
}


# --- Constants & State ---

VERSION = "1.5.0-beta"
START_TIME = datetime.utcnow()
APP_ROOT = ""
DOMAIN = ""

# In-memory state (will be loaded from DB)
zerodha_session = None

# --- Database for Persistence ---

DB_FILE = os.environ.get("DB_FILE_PATH", "state.db")

def init_db():
    """Initialize the SQLite database and create the settings table."""
    try:
        con = sqlite3.connect(DB_FILE)
        cur = con.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        con.commit()
        con.close()
        print("DB: Initialized successfully.")
    except Exception as e:
        print(f"DB: Error initializing database: {e}", file=sys.stderr)

def save_setting(key, value):
    """Save a key-value pair to the settings table. Value is stored as JSON."""
    try:
        con = sqlite3.connect(DB_FILE)
        cur = con.cursor()
        # Use INSERT OR REPLACE to handle both new and existing keys
        cur.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, json.dumps(value)))
        con.commit()
        con.close()
    except Exception as e:
        print(f"DB: Error saving setting '{key}': {e}", file=sys.stderr)

def get_setting(key, default=None):
    """Retrieve a setting by key. Value is parsed from JSON."""
    try:
        con = sqlite3.connect(DB_FILE)
        cur = con.cursor()
        cur.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = cur.fetchone()
        con.close()
        if row:
            return json.loads(row[0])
        return default
    except Exception as e:
        print(f"DB: Error getting setting '{key}': {e}", file=sys.stderr)
        return default

# --- App Factory ---

def create_app(testing=False):
	# --- App Setup ---
	app = Flask(__name__, static_folder="static", template_folder="templates")
	app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(16))
	app.config["SESSION_COOKIE_SECURE"] = not testing
	app.config["SESSION_COOKIE_HTTPONLY"] = True
	app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

    # --- Initialize DB and Load State ---
	init_db()
	global zerodha_session
	zerodha_session = get_setting("zerodha_session")
	if zerodha_session:
		print("State: Loaded Zerodha session from database.")

	# Zerodha config
	app.config["ZERODHA_API_KEY"] = _env("ZERODHA_API_KEY", "")
	app.config["ZERODHA_API_SECRET"] = _env("ZERODHA_API_SECRET", "")
	# Metrics auth token (optional). If set, /metrics requires Authorization: Bearer <token>
	app.config["METRICS_TOKEN"] = _env("METRICS_TOKEN", "")
	# Optional build/version metadata for support/ops
	app.config["APP_VERSION"] = _env("APP_VERSION", "dev")
	app.config["GIT_COMMIT"] = _env("GIT_COMMIT", "unknown")
	# Optional admin token to protect write APIs
	app.config["ADMIN_TOKEN"] = _env("ADMIN_TOKEN", "")
	# Optional maintenance mode (returns 503 to non-admins)
	app.config["MAINTENANCE_MODE"] = _env_bool("MAINTENANCE_MODE", False)
	# Use domain + optional base path for callback if set, else local dev callback
	domain = _env("DOMAIN", "")
	base_path = _env("APP_ROOT", _env("URL_PREFIX", "")).strip() or ""
	if base_path and not base_path.startswith("/"):
		base_path = "/" + base_path
	# Normalize trailing slash
	if base_path.endswith("/") and base_path != "/":
		base_path = base_path[:-1]
	default_callback = "http://127.0.0.1:5000/zerodha/callback"
	if domain and domain not in {"0.0.0.0", "127.0.0.1", "localhost"}:
		# Example: https://bambhoriaquantum.in/app/zerodha/callback
		app.config["ZERODHA_REDIRECT_URL"] = f"https://{domain}{base_path}/zerodha/callback"
	else:
		app.config["ZERODHA_REDIRECT_URL"] = default_callback

	# Session cookie hardening
	app.config.update(
		SESSION_COOKIE_HTTPONLY=True,
		SESSION_COOKIE_SAMESITE="Lax",
		SESSION_COOKIE_SECURE=bool(domain and domain not in {"0.0.0.0", "127.0.0.1", "localhost"}),
	)

	# App start time for uptime diagnostics
	app.config["START_TIME"] = int(time.time())

	# Make datetime available in Jinja templates (e.g., {{ datetime.utcnow().year }})
	@app.context_processor
	def _inject_globals():
		return {"datetime": datetime, "csp_nonce": getattr(g, "csp_nonce", None)}

	# Optional Sentry error tracking
	try:
		_sentry_dsn = _env("SENTRY_DSN", "")
		if _sentry_dsn:
			from sentry_sdk import init as sentry_init  # type: ignore
			from sentry_sdk.integrations.flask import FlaskIntegration  # type: ignore
			sentry_init(dsn=_sentry_dsn, integrations=[FlaskIntegration()], traces_sample_rate=float(_env("SENTRY_TRACES_SAMPLE", "0.0") or 0.0))
			# Log event for visibility
			# Note: _log_event defined later; use print here safely
			print("Sentry initialized", flush=True)
	except Exception:
		pass

	# Optional OpenTelemetry tracing (OTLP)
	try:
		otel_endpoint = _env("OTEL_EXPORTER_OTLP_ENDPOINT", "")
		if otel_endpoint:
			from opentelemetry import trace  # type: ignore
			from opentelemetry.instrumentation.flask import FlaskInstrumentor  # type: ignore
			from opentelemetry.instrumentation.requests import RequestsInstrumentor  # type: ignore
			from opentelemetry.sdk.resources import Resource  # type: ignore
			from opentelemetry.sdk.trace import TracerProvider  # type: ignore
			from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore
			from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # type: ignore

			service_name = _env("OTEL_SERVICE_NAME", "bambhoria-quantum")
			resource = Resource.create({
				"service.name": service_name,
				"service.version": _env("APP_VERSION", "dev"),
			})
			provider = TracerProvider(resource=resource)
			headers = _env("OTEL_EXPORTER_OTLP_HEADERS", "") or None
			exporter = OTLPSpanExporter(endpoint=otel_endpoint, headers=headers)
			processor = BatchSpanProcessor(exporter)
			provider.add_span_processor(processor)
			trace.set_tracer_provider(provider)
			FlaskInstrumentor().instrument_app(app)
			RequestsInstrumentor().instrument()
			app.config["OTEL_ENABLED"] = True
			print(f"OpenTelemetry initialized -> {otel_endpoint}", flush=True)
		else:
			app.config["OTEL_ENABLED"] = False
	except Exception as _otel_e:
		# Do not crash if OTEL packages are missing
		app.config["OTEL_ENABLED"] = False
		try:
			print(f"OpenTelemetry init skipped: {_otel_e}", flush=True)
		except Exception:
			pass

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

	# Global security headers and request id propagation
	@app.after_request
	def _set_security_headers(resp):
		try:
			resp.headers.setdefault("X-Content-Type-Options", "nosniff")
			resp.headers.setdefault("X-Frame-Options", "DENY")
			resp.headers.setdefault("Referrer-Policy", "no-referrer")
			resp.headers.setdefault("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
			# Content-Security-Policy with per-request nonce for inline scripts; allow Google Fonts
			nonce = getattr(g, "csp_nonce", None)
			csp_parts = [
				"default-src 'self'",
				"img-src 'self' data:",
				"style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
				"font-src 'self' https://fonts.gstatic.com",
			]
			if nonce:
				csp_parts.append(f"script-src 'self' 'nonce-{nonce}'")
			else:
				csp_parts.append("script-src 'self'")
			resp.headers.setdefault("Content-Security-Policy", "; ".join(csp_parts))
			# advise HTTPS-only if behind TLS (safe to send always)
			resp.headers.setdefault("Strict-Transport-Security", "max-age=63072000; includeSubDomains; preload")
			# avoid caching sensitive API responses globally
			resp.headers.setdefault("Cache-Control", "no-store")
			# propagate a request id if available
			req_id = getattr(g, "request_id", None)
			if req_id:
				resp.headers.setdefault("X-Request-Id", req_id)
		except Exception:
			pass
		return resp

	@app.before_request
	def _assign_request_id():
		# capture or create a per-request id for correlation
		try:
			req_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
			g.request_id = req_id
		except Exception:
			pass

	# Generate a per-request CSP nonce for secure inline scripts
	@app.before_request
	def _assign_csp_nonce():
		try:
			g.csp_nonce = secrets.token_urlsafe(16)
		except Exception:
			g.csp_nonce = None

	# Maintenance mode gate: block non-admin when enabled (allow health/metrics/docs)
	@app.before_request
	def _maintenance_gate():
		try:
			if not app.config.get("MAINTENANCE_MODE"):
				return None
			# allowed paths
			path = request.path or "/"
			allowed = {"/health", "/live", "/ready", "/version", "/metrics", "/robots.txt", "/openapi.json", "/docs"}
			if path in allowed or path.startswith("/.well-known"):
				return None
			# admin bypass
			token = app.config.get("ADMIN_TOKEN", "")
			h = request.headers.get("Authorization", "")
			if token and h.startswith("Bearer ") and h.split(" ", 1)[1].strip() == token:
				return None
			return Response("Service under maintenance. Please try again later.\n", status=503, mimetype="text/plain")
		except Exception:
			return None

	# Lightweight in-memory rate limiter (per-IP, per-endpoint)
	app.config["_RATE_LIMITS"] = {}

	def _rate_limit(limit: int, window: int = 60):
		"""Decorator to limit requests per window seconds by client IP and endpoint.
		Bypasses if valid ADMIN_TOKEN bearer is provided.
		"""
		def decorator(fn):
			@functools.wraps(fn)
			def wrapper(*args, **kwargs):
				try:
					# Admin bearer bypass
					token = app.config.get("ADMIN_TOKEN", "")
					h = request.headers.get("Authorization", "")
					if token and h.startswith("Bearer ") and h.split(" ", 1)[1].strip() == token:
						return fn(*args, **kwargs)
					key = f"{request.remote_addr}:{fn.__name__}"
					now = time.time()
					bucket = app.config["_RATE_LIMITS"].setdefault(key, [])
					# drop old
					threshold = now - window
					bucket[:] = [t for t in bucket if t > threshold]
					if len(bucket) >= limit:
						retry_after = int(bucket[0] - threshold) + 1 if bucket else window
						# increment metric on rate-limit
						try:
							if _metrics.get("rate_limited_total"):
								_metrics["rate_limited_total"].inc()  # type: ignore
						except Exception:
							pass
						resp = jsonify({"ok": False, "error": "rate_limited", "retry_after": retry_after})
						resp.status_code = 429
						resp.headers["Retry-After"] = str(retry_after)
						return resp
					bucket.append(now)
				except Exception:
					pass
				return fn(*args, **kwargs)
			return wrapper
		return decorator

	def _require_admin() -> Optional[Response]:
		"""Optional bearer-token check for state-changing endpoints.
		If ADMIN_TOKEN is unset, allow by default for backwards compatibility.
		"""
		token = app.config.get("ADMIN_TOKEN", "")
		if not token:
			return None
		h = request.headers.get("Authorization", "")
		if not h.startswith("Bearer ") or h.split(" ", 1)[1].strip() != token:
			return jsonify({"ok": False, "error": "unauthorized"}), 401
		return None

	# Initialize Prometheus metrics (if available)
	if METRICS_ENABLED:
		try:
			_metrics["events_total"] = Counter("app_events_total", "Events logged", ["level", "type"])  # type: ignore
			_metrics["live_running"] = Gauge("app_live_running", "Live runner running (1/0)")  # type: ignore
			_metrics["demo_mode"] = Gauge("app_demo_mode", "Demo mode enabled (1/0)")  # type: ignore
			_metrics["auto_heal"] = Gauge("app_auto_heal", "Auto-heal enabled (1/0)")  # type: ignore
			_metrics["authenticated"] = Gauge("app_authenticated", "Authenticated with Zerodha (1/0)")  # type: ignore
			_metrics["last_beat"] = Gauge("app_last_beat", "Epoch seconds of last runner beat")  # type: ignore
			_metrics["quotes_success_total"] = Counter("app_quotes_success_total", "Quotes fetch successes")  # type: ignore
			_metrics["quotes_error_total"] = Counter("app_quotes_error_total", "Quotes fetch errors")  # type: ignore
			_metrics["profile_success_total"] = Counter("app_profile_success_total", "Profile fetch successes")  # type: ignore
			_metrics["profile_error_total"] = Counter("app_profile_error_total", "Profile fetch errors")  # type: ignore
			_metrics["rate_limited_total"] = Counter("app_rate_limited_total", "Requests rate-limited")  # type: ignore
			_metrics["panic_total"] = Counter("app_panic_total", "Panic stop invoked")  # type: ignore
		except Exception:
			# If metrics init fails, continue without metrics
			pass

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
		# increment metrics counter (best-effort)
		try:
			if _metrics.get("events_total"):
				_metrics["events_total"].labels(level=level, type=etype).inc()  # type: ignore
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
				app.config["ZERODHA_SESSION"] = json.load(f)
	except Exception:
		pass

	@app.route("/")
	def index():
		return render_template("index.html")

	@app.route("/health")
	def health():
		# keep gauges roughly in sync on health checks
		try:
			run = app.config.get("LIVE_RUNNER", {})
			if _metrics.get("live_running"):
				_metrics["live_running"].set(1.0 if run.get("running") else 0.0)  # type: ignore
			if _metrics.get("demo_mode"):
				_metrics["demo_mode"].set(1.0 if run.get("demo_mode") else 0.0)  # type: ignore
			if _metrics.get("auto_heal"):
				_metrics["auto_heal"].set(1.0 if run.get("auto_heal", True) else 0.0)  # type: ignore
			sess = app.config.get("ZERODHA_SESSION")
			if _metrics.get("authenticated"):
				_metrics["authenticated"].set(1.0 if (sess and sess.get("access_token")) else 0.0)  # type: ignore
		except Exception:
			pass
		return {"status": "ok", "app": "bambhoria-quantum", "version": "live"}

	@app.route("/live")
	def live():
		# Liveness: process is up
		return jsonify({"ok": True, "live": True, "ts": int(time.time())})

	@app.route("/ready")
	def ready():
		# Readiness: app config loaded and runner supervisor thread started
		try:
			run = app.config.get("LIVE_RUNNER", {})
			sess = app.config.get("ZERODHA_SESSION")
			resp = {
				"ok": True,
				"ready": True,
				"authenticated": bool(sess and sess.get("access_token")),
				"demo_mode": bool(run.get("demo_mode")),
				"auto_heal": bool(run.get("auto_heal", True)),
			}
			return jsonify(resp)
		except Exception as e:
			return jsonify({"ok": False, "ready": False, "error": str(e)}), 500

	@app.route("/version")
	def version():
		return jsonify({
			"app": "bambhoria-quantum",
			"version": app.config.get("APP_VERSION", "dev"),
			"commit": app.config.get("GIT_COMMIT", "unknown"),
			"time": datetime.utcnow().isoformat(timespec="seconds") + "Z",
		})

	@app.route("/metrics")
	def metrics():
		# optional bearer protection
		token = app.config.get("METRICS_TOKEN", "")
		if token:
			h = request.headers.get("Authorization", "")
			if not h.startswith("Bearer ") or h.split(" ", 1)[1].strip() != token:
				return Response("unauthorized\n", status=401, mimetype="text/plain")
		if METRICS_ENABLED and generate_latest:
			try:
				return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)
			except Exception:
				# fall back to simple text when generation fails
				return Response("metric_export_error 1\n", mimetype="text/plain")
		return Response("metrics_disabled 1\n", mimetype="text/plain")

	# ---------- Zerodha OAuth ----------
	@app.route("/login/zerodha")
	@_rate_limit(20, 60)
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
	@_rate_limit(60, 60)
	def zerodha_callback():
		global zerodha_session
		request_token = request.args.get("request_token")
		if not request_token:
			return "Error: No request_token provided.", 400

		try:
			# Exchange request_token for an access_token
			api_key = app.config["ZERODHA_API_KEY"]
			api_secret = os.environ.get("ZERODHA_CLIENT_SECRET")
			kite = KiteConnect(api_key=api_key)
			session_data = kite.generate_session(request_token, api_secret=api_secret)

			# Persist session
			zerodha_session = session_data
			save_setting("zerodha_session", zerodha_session)
			print(f"State: Saved new Zerodha session for user {session_data.get('user_id')}")


			# Store minimal info in Flask session for frontend
			session["user_id"] = session_data.get("user_id")
			session["access_token"] = session_data.get("access_token")
			session["refresh_token"] = session_data.get("refresh_token")
			session["expires_at"] = session_data.get("expires_at")

			_log_event("info", "auth.callback", "Zerodha login successful.", {"user_id": session_data.get("user_id")})

			# metrics: set authenticated gauge
			try:
				if _metrics.get("authenticated"):
					_metrics["authenticated"].set(1.0 if session_data.get("access_token") else 0.0)  # type: ignore
			except Exception:
				pass

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

	def _get_quotes(instruments: List[str]) -> Dict[str, Any]:
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
			try:
				if _metrics.get("profile_success_total"):
					_metrics["profile_success_total"].inc()  # type: ignore
				if _metrics.get("last_beat"):
					_metrics["last_beat"].set(run["last_beat"])  # type: ignore
				if _metrics.get("live_running"):
					_metrics["live_running"].set(1.0)  # type: ignore
			except Exception:
				pass
			_log_event("info", "runner", "Verified profile; entering live quote loop")
			# Poll quotes every 5s
			while run.get("running"):
				try:
					quotes = _get_quotes(run.get("symbols", []))
					run["last_quotes"] = quotes
					run["last_error"] = None
					run["last_beat"] = int(time.time())
					try:
						if _metrics.get("quotes_success_total"):
							_metrics["quotes_success_total"].inc()  # type: ignore
						if _metrics.get("last_beat"):
							_metrics["last_beat"].set(run["last_beat"])  # type: ignore
					except Exception:
						pass
				except Exception as ie:
					run["last_error"] = str(ie)
					_log_event("error", "quotes", f"Quote fetch error: {ie}")
					try:
						if _metrics.get("quotes_error_total"):
							_metrics["quotes_error_total"].inc()  # type: ignore
					except Exception:
						pass
				time.sleep(5)
		except Exception as e:
			run["last_error"] = str(e)
			_log_event("error", "runner", f"Live loop error: {e}")
			try:
				if _metrics.get("profile_error_total"):
					_metrics["profile_error_total"].inc()  # type: ignore
			except Exception:
				pass
		finally:
			run["running"] = False
			_log_event("warn", "runner", "Live runner stopped")
			try:
				if _metrics.get("live_running"):
					_metrics["live_running"].set(0.0)  # type: ignore
			except Exception:
				pass

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
	@_rate_limit(30, 60)
	def api_set_symbols():
		run = app.config.get("LIVE_RUNNER", {})
		if request.method == "POST":
			_guard = _require_admin()
			if _guard is not None:
				return _guard
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
	@_rate_limit(30, 60)
	def api_start_trading():
		_guard = _require_admin()
		if _guard is not None:
			return _guard
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
	@_rate_limit(30, 60)
	def api_stop_trading():
		_guard = _require_admin()
		if _guard is not None:
			return _guard
		run = app.config.get("LIVE_RUNNER", {})
		run["running"] = False
		_log_event("info", "runner", "Stop requested")
		return jsonify({"ok": True, "message": "Live trading loop stop requested."})

	@app.route("/api/toggle_demo", methods=["POST"]) 
	@_rate_limit(30, 60)
	def api_toggle_demo():
		_guard = _require_admin()
		if _guard is not None:
			return _guard
		run = app.config.get("LIVE_RUNNER", {})
		try:
			payload = request.get_json(silent=True) or {}
			enable = bool(payload.get("enable"))
			run["demo_mode"] = enable
			_log_event("info", "settings", f"Demo mode set to {enable}")
			try:
				if _metrics.get("demo_mode"):
					_metrics["demo_mode"].set(1.0 if enable else 0.0)  # type: ignore
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
			return jsonify({"ok": True, "demo_mode": run["demo_mode"]})
		except Exception as e:
			return jsonify({"ok": False, "message": str(e)}), 400

	@app.route("/api/toggle_autoheal", methods=["POST"]) 
	@_rate_limit(30, 60)
	def api_toggle_autoheal():
		_guard = _require_admin()
		if _guard is not None:
			return _guard
		run = app.config.get("LIVE_RUNNER", {})
		try:
			payload = request.get_json(silent=True) or {}
			enable = bool(payload.get("enable"))
			run["auto_heal"] = enable
			_log_event("info", "settings", f"Auto-heal set to {enable}")
			try:
				if _metrics.get("auto_heal"):
					_metrics["auto_heal"].set(1.0 if enable else 0.0)  # type: ignore
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

	def _sse_encode(data: str, event: Optional[str] = None, id: Optional[str] = None) -> str:
		lines = []
		if event:
			lines.append(f"event: {event}")
		if id:
			lines.append(f"id: {id}")
		for line in data.splitlines():
			lines.append(f"data: {line}")
		return "\n".join(lines) + "\n\n"

	@app.route("/stream/events")
	def stream_events():
		# Admin-only: events can carry sensitive info
		_guard = _require_admin()
		if _guard is not None:
			return _guard
		def generate():
			last_idx = 0
			while True:
				try:
					buf = app.config.get("EVENTS", [])
					if last_idx < len(buf):
						for i in range(last_idx, len(buf)):
							payload = json.dumps(buf[i], ensure_ascii=False)
							yield _sse_encode(payload, event="event", id=str(buf[i].get("ts", i)))
						last_idx = len(buf)
					yield _sse_encode(json.dumps({"ok": True, "ping": int(time.time())}), event="ping")
					time.sleep(2)
				except GeneratorExit:
					break
				except Exception as e:
					# backoff a bit on error
					time.sleep(2)
		resp = Response(stream_with_context(generate()), mimetype="text/event-stream")
		resp.headers["Cache-Control"] = "no-cache"
		resp.headers["X-Accel-Buffering"] = "no"
		return resp

	@app.route("/stream/quotes")
	def stream_quotes():
		def generate():
			last_beat = None
			while True:
				try:
					run = app.config.get("LIVE_RUNNER", {})
					beat = run.get("last_beat")
					if beat and beat != last_beat:
						payload = json.dumps(run.get("last_quotes") or {}, ensure_ascii=False)
						yield _sse_encode(payload, event="quotes", id=str(beat))
						last_beat = beat
					yield _sse_encode(json.dumps({"ok": True, "ping": int(time.time())}), event="ping")
					time.sleep(2)
				except GeneratorExit:
					break
				except Exception:
					time.sleep(2)
		resp = Response(stream_with_context(generate()), mimetype="text/event-stream")
		resp.headers["Cache-Control"] = "no-cache"
		resp.headers["X-Accel-Buffering"] = "no"
		return resp

	@app.route("/api/panic_stop", methods=["POST"]) 
	@_rate_limit(10, 60)
	def api_panic_stop():
		_guard = _require_admin()
		if _guard is not None:
			return _guard
		try:
			run = app.config.get("LIVE_RUNNER", {})
			run["running"] = False
			run["auto_heal"] = False
			_log_event("critical", "panic", "PANIC STOP invoked: runner stopped and auto-heal disabled")
			try:
				if _metrics.get("panic_total"):
					_metrics["panic_total"].inc()  # type: ignore
			except Exception:
				pass
			return jsonify({"ok": True, "message": "PANIC: stopped and disabled auto-heal"})
		except Exception as e:
			return jsonify({"ok": False, "error": str(e)}), 500

	@app.route("/api/toggle_maintenance", methods=["POST"])
	@_rate_limit(10, 60)
	def api_toggle_maintenance():
		_guard = _require_admin()
		if _guard is not None:
			return _guard
		try:
			payload = request.get_json(silent=True) or {}
			enable = bool(payload.get("enable"))
			app.config["MAINTENANCE_MODE"] = enable
			return jsonify({"ok": True, "maintenance_mode": enable})
		except Exception as e:
			return jsonify({"ok": False, "error": str(e)}), 400
	@app.route("/diagnostics")
	def diagnostics():
		# Admin-only diagnostics to avoid leaking details publicly
		_guard = _require_admin()
		if _guard is not None:
			return _guard
		try:
			run = app.config.get("LIVE_RUNNER", {})
			sess = app.config.get("ZERODHA_SESSION") or {}
			uptime = int(time.time()) - int(app.config.get("START_TIME", int(time.time())))
			probe = request.args.get("probe") == "1"
			kite_probe = None
			if probe:
				try:
					pr = requests.get("https://api.kite.trade/", timeout=5)
					kite_probe = {"ok": True, "status": pr.status_code}
				except Exception as e:
					kite_probe = {"ok": False, "error": str(e)}
			resp = {
				"ok": True,
				"time_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
				"uptime_seconds": uptime,
				"python": sys.version.split(" ")[0],
				"platform": platform.platform(),
				"pid": os.getpid(),
				"threads": threading.active_count(),
				"domain": domain,
				"app_root": base_path,
				"redirect_url": app.config.get("ZERODHA_REDIRECT_URL"),
				"metrics_enabled": bool(METRICS_ENABLED),
				"admin_protection": bool(app.config.get("ADMIN_TOKEN")),
				"otel_enabled": bool(app.config.get("OTEL_ENABLED", False)),
				"authenticated": bool(sess.get("access_token")),
				"runner": {
					"running": bool(run.get("running")),
					"demo_mode": bool(run.get("demo_mode")),
					"auto_heal": bool(run.get("auto_heal", True)),
					"last_beat": run.get("last_beat"),
					"last_error": run.get("last_error"),
					"symbols_count": len(run.get("symbols", [])),
				},
				"env_summary": {
					"ZERODHA_API_KEY": bool(app.config.get("ZERODHA_API_KEY")),
					"ZERODHA_API_SECRET": bool(app.config.get("ZERODHA_API_SECRET")),
					"SECRET_KEY": bool(_env("FLASK_SECRET_KEY") or _env("SECRET_KEY")),
					"DOMAIN": bool(domain),
					"APP_ROOT": bool(base_path),
					"METRICS_TOKEN": bool(app.config.get("METRICS_TOKEN")),
					"ADMIN_TOKEN": bool(app.config.get("ADMIN_TOKEN")),
				},
			}
			if probe:
				resp["probes"] = {"kite_trade_home": kite_probe}
			return jsonify(resp)
		except Exception as e:
			return jsonify({"ok": False, "error": str(e)}), 500

	@app.route("/robots.txt")
	def robots_txt():
		return Response("User-agent: *\nDisallow: /\n", mimetype="text/plain")

	@app.route("/.well-known/security.txt")
	def security_txt():
		try:
			contact = f"mailto:security@{domain}" if domain else "mailto:security@example.com"
			body = f"Contact: {contact}\nPolicy: https://{domain}/security\nPreferred-Languages: en\n" if domain else f"Contact: {contact}\nPreferred-Languages: en\n"
			return Response(body, mimetype="text/plain")
		except Exception:
			return Response("Contact: mailto:security@example.com\nPreferred-Languages: en\n", mimetype="text/plain")

	@app.route("/openapi.json")
	def openapi_spec():
		# Minimal OpenAPI spec for core endpoints
		server_url = (f"https://{domain}{base_path}" if domain else f"http://127.0.0.1:5000")
		spec = {
			"openapi": "3.0.0",
			"info": {"title": "Bambhoria Quantum API", "version": app.config.get("APP_VERSION", "dev")},
			"servers": [{"url": server_url}],
			"paths": {
				"/health": {"get": {"summary": "Health", "responses": {"200": {"description": "OK"}}}},
				"/live": {"get": {"summary": "Liveness", "responses": {"200": {"description": "OK"}}}},
				"/ready": {"get": {"summary": "Readiness", "responses": {"200": {"description": "OK"}}}},
				"/version": {"get": {"summary": "Version", "responses": {"200": {"description": "OK"}}}},
				"/metrics": {"get": {"summary": "Prometheus metrics", "responses": {"200": {"description": "OK"}, "401": {"description": "Unauthorized"}}}},
				"/login/zerodha": {"get": {"summary": "Redirect to Zerodha login", "responses": {"302": {"description": "Redirect"}}}},
				"/zerodha/callback": {"get": {"summary": "Zerodha OAuth callback", "parameters": [{"in": "query", "name": "request_token", "schema": {"type": "string"}}], "responses": {"302": {"description": "Redirect on success"}, "400": {"description": "Bad Request"}, "500": {"description": "Server Error"}}}},
				"/api/status": {"get": {"summary": "App status", "responses": {"200": {"description": "OK"}}}},
				"/api/events": {"get": {"summary": "Recent events", "parameters": [{"in": "query", "name": "limit", "schema": {"type": "integer"}}], "responses": {"200": {"description": "OK"}}}},
				"/api/logs/events/download": {"get": {"summary": "Download events log", "responses": {"200": {"description": "File"}, "404": {"description": "Not Found"}}}},
				"/api/set_symbols": {"post": {"summary": "Set symbols", "responses": {"200": {"description": "OK"}, "401": {"description": "Unauthorized"}, "400": {"description": "Bad Request"}}}},
				"/api/start_trading": {"post": {"summary": "Start trading", "responses": {"200": {"description": "OK"}, "401": {"description": "Unauthorized"}}}},
				"/api/stop_trading": {"post": {"summary": "Stop trading", "responses": {"200": {"description": "OK"}, "401": {"description": "Unauthorized"}}}},
				"/api/toggle_demo": {"post": {"summary": "Toggle demo mode", "responses": {"200": {"description": "OK"}, "401": {"description": "Unauthorized"}}}},
				"/api/toggle_autoheal": {"post": {"summary": "Toggle auto-heal", "responses": {"200": {"description": "OK"}, "401": {"description": "Unauthorized"}}}},
				"/api/live_status": {"get": {"summary": "Live status", "responses": {"200": {"description": "OK"}}}},
				"/api/panic_stop": {"post": {"summary": "PANIC stop (admin)", "responses": {"200": {"description": "OK"}, "401": {"description": "Unauthorized"}}}},
				"/stream/events": {"get": {"summary": "Event stream (SSE, admin)", "responses": {"200": {"description": "text/event-stream"}, "401": {"description": "Unauthorized"}}}},
				"/stream/quotes": {"get": {"summary": "Quotes stream (SSE)", "responses": {"200": {"description": "text/event-stream"}}}},
				"/diagnostics": {"get": {"summary": "Admin diagnostics", "responses": {"200": {"description": "OK"}, "401": {"description": "Unauthorized"}}}},
			}
		}
		return jsonify(spec)

	@app.route("/docs")
	def docs():
		# Minimal Redoc page; override CSP to allow CDN for this page only
		html = f"""
		<!DOCTYPE html>
		<html>
		<head>
			<meta charset=\"utf-8\"/>
			<title>Bambhoria Quantum API Docs</title>
			<link rel=\"preconnect\" href=\"https://cdn.jsdelivr.net\"> 
		</head>
		<body>
			<redoc spec-url=\"{base_path or ''}/openapi.json\"></redoc>
			<script src=\"https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js\"></script>
		</body>
		</html>
		"""
		resp = Response(html, mimetype="text/html")
		resp.headers["Content-Security-Policy"] = "default-src 'self' https://cdn.jsdelivr.net; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self' https://cdn.jsdelivr.net"
		return resp

	# Error handlers
	@app.errorhandler(429)
	def _rate_limited(err):
		try:
			retry_after = getattr(err, "retry_after", None) or request.headers.get("Retry-After") or "60"
		except Exception:
			retry_after = "60"
		resp = jsonify({"ok": False, "error": "rate_limited", "request_id": getattr(g, "request_id", None)})
		resp.status_code = 429
		resp.headers["Retry-After"] = str(retry_after)
		return resp

	@app.errorhandler(Exception)
	def _unhandled_error(err):
		try:
			# log a critical event with correlation id
			_log_event("critical", "unhandled", f"Unhandled error: {err}", {"request_id": getattr(g, "request_id", None), "path": request.path})
		except Exception:
			pass
		resp = jsonify({"ok": False, "error": "internal_error", "request_id": getattr(g, "request_id", None)})
		resp.status_code = 500
		return resp

	return app


# Flask app instance for WSGI
app = create_app()

if __name__ == "__main__":
	app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)

