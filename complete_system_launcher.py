from __future__ import annotations
import sys
try:
    from complete_system_launcher_v2 import main as _main
except Exception as e:  # pragma: no cover - only if v2 missing
    def main() -> int:
        print("[Bambhoria] Could not import complete_system_launcher_v2.main:", e)
        print("[Bambhoria] Please run: python complete_system_launcher_v2.py")
        return 1
else:
    def main() -> int:
        return _main()

if __name__ == "__main__":
    sys.exit(main())

'''  # BEGIN LEGACY CONTENT (ignored)
"""
complete_system_launcher.py
Bambhoria Complete System with Quantum Performance Monitoring
Ultimate launcher for the complete trading ecosystem
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil
import requests


class BambhoriaSystemLauncher:
    def __init__(self, no_open: bool = False, no_watchdog: bool = False):
        self.processes = []
            self.no_open = no_open
            self.no_watchdog = no_watchdog
            self._running = True
            self._watchdog_thread = None

            # Ensure logs directory exists for process outputs
            self.logs_dir = Path("logs")
            try:
                self.logs_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

            # Load ports/endpoints configuration with environment overrides
            self.ports_cfg = self._load_ports_config()

            # Define services. Only those with a 'port' are web dashboards we can probe.
            self.services = {
                "System Monitor": {
                    "script": "system_monitor_dashboard.py",
                    "port": self.ports_cfg["system_monitor"],
                    "delay": 3,
                },
                "Analytics Dashboard": {
                    "script": "dashboard_server.py",
                    "port": self.ports_cfg["analytics_dashboard"],
                    "delay": 3,
                },
                "Performance Monitor": {
                    "script": "performance_dashboard.py",
                    "port": self.ports_cfg["performance_monitor"],
                    "delay": 2,
                },
                # http_mock_feed is a client that posts to an API; it does not expose an HTTP port
                "Mock Feed Client": {
                    "script": "http_mock_feed.py",
                    "port": None,
                    "delay": 3,
                    "args": ["--url", self.ports_cfg["mock_feed_api"]],
                },
                "Complete Trading System": {
                    "script": "complete_trading_system.py",
                    "port": None,
                    "delay": 5,
                },
            }

            # Background monitor (no watchdog restarts for this one)
            self.background_services = {
                "Quantum Monitor": {"script": "quantum_performance_monitor.py", "port": None}
            }

        def _load_ports_config(self) -> dict:
            """Load ports/endpoints from config/ports.json with environment overrides."""
            cfg_path = Path("config/ports.json")
            data = {
                "system_monitor": 5008,
                "analytics_dashboard": 5000,
                "performance_monitor": 5009,
                "mock_feed_api": "http://localhost:5002/api/ticks",
            }
            try:
                if cfg_path.exists():
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                        if isinstance(loaded, dict):
                            for k in data.keys():
                                if k in loaded:
                                    data[k] = loaded[k]
            except Exception:
                # Fall back to defaults on any error
                pass

            # Environment overrides (if set)
            def _int_env(name: str, default: int) -> int:
                try:
                    v = os.getenv(name)
                    return int(v) if v is not None else default
                except Exception:
                    return default

            data["system_monitor"] = _int_env("BAMBHORIA_PORT_SYSTEM_MONITOR", data["system_monitor"])
            data["analytics_dashboard"] = _int_env("BAMBHORIA_PORT_ANALYTICS", data["analytics_dashboard"])
            data["performance_monitor"] = _int_env("BAMBHORIA_PORT_PERF", data["performance_monitor"])
            api_env = os.getenv("BAMBHORIA_MOCK_FEED_API")
            if api_env:
                data["mock_feed_api"] = api_env

            return data

        def print_banner(self):
            banner = (
                "\n"
                "╔══════════════════════════════════════════════════════════════════════════════╗\n"
                "║                   🎯 BAMBHORIA COMPLETE TRADING SYSTEM 🎯                   ║\n"
                "║                         with Quantum Performance Monitoring                  ║\n"
                "╠══════════════════════════════════════════════════════════════════════════════╣\n"
                "║  🚀 AI-Powered Trading Pipeline                                             ║\n"
                "║  📊 Real-time Performance Monitoring                                        ║\n"
                "║  🛡️  Advanced Risk Management                                               ║\n"
                "║  💎 Multi-Dashboard Visualization                                           ║\n"
                "║  ⚡ Quantum System Health Monitoring                                        ║\n"
                "╚══════════════════════════════════════════════════════════════════════════════╝\n"
            )
            print(banner)

        def check_port_status(self, port: int, timeout: float = 3.0) -> bool:
            """Check if a port is open and responding on localhost."""
            try:
                response = requests.get(f"http://localhost:{port}", timeout=timeout)
                return response.status_code in (200, 404)  # 404 is acceptable for API endpoints
            except Exception:
                return False

        def start_service(self, name: str, config: dict):
            """Start a single service and stream its logs to a file."""
            script = config["script"]
            port = config.get("port")
            delay = config.get("delay", 2)
            extra_args = config.get("args", [])

            print(f"🚀 Starting {name}...")

            # Check if port is already serving something; if so, skip starting a server on it
            if port and self.check_port_status(port, timeout=1):
                print(f"⚠️  Port {port} already in use for {name}")
                return None

            try:
                # Stream output to log file to avoid PIPE buffer deadlocks
                log_path = self.logs_dir / f"{Path(script).stem}.log"
                log_file = open(log_path, "a", encoding="utf-8", buffering=1)
                argv = [sys.executable, script, *extra_args]
                process = subprocess.Popen(
                    argv,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=os.getcwd(),
                )

                time.sleep(delay)

                # Check if process is still running
                if process.poll() is None:
                    print(f"✅ {name} started successfully")
                    return {
                        "name": name,
                        "process": process,
                        "port": port,
                        "config": config,
                        "restarts": 0,
                    }
                else:
                    print(f"❌ {name} failed to start")
                    return None

            except Exception as e:
                print(f"❌ Error starting {name}: {e}")
                return None

        def start_background_monitor(self):
            """Start quantum performance monitor in background."""
            try:
                log_path = self.logs_dir / "quantum_performance_monitor.log"
                log_file = open(log_path, "a", encoding="utf-8", buffering=1)
                quantum_process = subprocess.Popen(
                    [sys.executable, "quantum_performance_monitor.py"],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=os.getcwd(),
                )

                print("⚡ Quantum Performance Monitor running in background")
                return quantum_process
            except Exception as e:
                print(f"⚠️  Could not start Quantum Monitor: {e}")
                return None

        def open_dashboards(self):
            """Open all dashboards in a browser unless headless mode is enabled."""
            if self.no_open:
                print("\n🖥️  Headless mode (--no-open): skipping opening dashboards in browser")
                return

            print("\n🌐 Opening dashboards in browser...")
            dashboard_urls = [
                f"http://localhost:{self.ports_cfg['system_monitor']}",  # System Monitor
                f"http://localhost:{self.ports_cfg['analytics_dashboard']}",  # Analytics Dashboard
                f"http://localhost:{self.ports_cfg['performance_monitor']}",  # Performance Monitor
            ]

            for url in dashboard_urls:
                try:
                    # Windows 'start' must be invoked via shell=True
                    subprocess.run(["start", url], shell=True, check=False)
                    time.sleep(1)
                except Exception:
                    print(f"Could not open {url}")

        def _watchdog_loop(self):
            """Background watchdog: restarts failed services with exponential backoff."""
            MAX_RESTARTS = 5
            while self._running:
                try:
                    for i, svc in enumerate(list(self.processes)):
                        name = svc.get("name")
                        if name == "Quantum Monitor":
                            # We do not restart the background monitor automatically
                            continue
                        proc = svc.get("process")
                        if not proc:
                            continue
                        # Process exited?
                        if proc.poll() is not None:
                            cfg = svc.get("config", {})
                            restarts = svc.get("restarts", 0)
                            if restarts >= MAX_RESTARTS:
                                print(f"❌ {name} exceeded max restarts; not restarting")
                                continue
                            backoff = min(2 ** restarts, 30)
                            print(
                                f"⚠️  {name} exited. Restarting in {backoff}s (attempt {restarts + 1}/{MAX_RESTARTS})"
                            )
                            time.sleep(backoff)
                            # Attempt restart
                            new_info = self.start_service(name, cfg)
                            if new_info:
                                new_info["restarts"] = restarts + 1
                                self.processes[i] = new_info
                                print(f"✅ {name} restarted")
                            else:
                                svc["restarts"] = restarts + 1
                                self.processes[i] = svc
                    time.sleep(2)
                except Exception:
                    time.sleep(2)

        def start_watchdog(self):
            if self.no_watchdog:
                print("🛡️  Watchdog disabled (--no-watchdog)")
                return
            if self._watchdog_thread and self._watchdog_thread.is_alive():
                return
            print("🛡️  Starting watchdog thread")
            self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
            self._watchdog_thread.start()

        def health_check(self) -> bool:
            """Perform comprehensive health check.

            Returns True if overall health looks OK, False otherwise.
            """
            print("\n🔍 System Health Check:")
            print("-" * 50)

            # Check services that expose ports
            healthy_services = 0
            total_services = sum(1 for _n, cfg in self.services.items() if cfg.get("port"))

            for name, config in self.services.items():
                port = config.get("port")
                if port:
                    if self.check_port_status(port):
                        print(f"✅ {name} (Port {port}): Healthy")
                        healthy_services += 1
                    else:
                        print(f"❌ {name} (Port {port}): Not responding")

            # System resource check
            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage(".")

            print(f"\n💻 System Resources:")
            print(f"   CPU Usage: {cpu}%")
            print(f"   Memory Usage: {memory.percent}%")
            print(f"   Disk Usage: {round(disk.percent, 1)}%")
            print(f"   Available Memory: {round(memory.available / (1024**3), 2)} GB")

            # Overall health score
            service_health = (healthy_services / total_services) * 100 if total_services > 0 else 0
            resource_health = max(0, 100 - max(cpu - 50, memory.percent - 60, disk.percent - 70))
            overall_health = (service_health + resource_health) / 2

            print(f"\n🎯 Overall System Health: {overall_health:.1f}%")

            return overall_health > 70

        def monitor_system_runtime(self):
            """Monitor system during runtime."""
            print("\n📊 Real-time System Monitoring:")
            print("=" * 60)

            start_time = datetime.now()

            # Start watchdog in background
            self.start_watchdog()

            try:
                while True:
                    current_time = datetime.now()
                    uptime = current_time - start_time

                    # Get current metrics
                    cpu = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()

                    # Check service health
                    total_services = sum(1 for _n, cfg in self.services.items() if cfg.get("port"))
                    active_services = 0
                    for _name, config in self.services.items():
                        port = config.get("port")
                        if port and self.check_port_status(port, timeout=1):
                            active_services += 1

                    # Display status
                    status_line = (
                        f"[{current_time.strftime('%H:%M:%S')}] "
                        f"CPU: {cpu:5.1f}% | "
                        f"MEM: {memory.percent:5.1f}% | "
                        f"Services: {active_services}/{total_services} | "
                        f"Uptime: {str(uptime).split('.')[0]}"
                    )

                    print(f"\r{status_line}", end="", flush=True)

                    time.sleep(5)

            except KeyboardInterrupt:
                print("\n\n🛑 Shutting down system...")

        def shutdown_all(self):
            """Gracefully shutdown all services."""
            print("\n🛑 Shutting down all services...")

            self._running = False
            try:
                if self._watchdog_thread:
                    self._watchdog_thread.join(timeout=1)
            except Exception:
                pass

            for service_info in self.processes:
                if service_info and service_info.get("process"):
                    name = service_info.get("name", "(unknown)")
                    process = service_info["process"]

                    print(f"   Stopping {name}...")
                    process.terminate()

                    # Wait for graceful shutdown
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()

            print("✅ All services stopped successfully!")

        def launch_complete_system(self):
            """Launch the complete Bambhoria trading system."""
            self.print_banner()

            print(f"📂 Working Directory: {os.getcwd()}")
            print(f"🐍 Python: {sys.executable}")
            print(f"⏰ Launch Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            try:
                # Start all main services
                print("\n🚀 Starting Core Services:")
                print("=" * 40)

                for name, config in self.services.items():
                    service_info = self.start_service(name, config)
                    if service_info:
                        self.processes.append(service_info)

                # Start background quantum monitor
                quantum_process = self.start_background_monitor()
                if quantum_process:
                    self.processes.append(
                        {"name": "Quantum Monitor", "process": quantum_process, "port": None}
                    )

                # Wait for services to stabilize
                print("\n⏳ Waiting for services to initialize...")
                time.sleep(8)

                # Perform health check
                if self.health_check():
                    print("\n🎉 System launched successfully!")

                    # Show access information
                    print(f"\n📊 Access Your Dashboards:")
                    print(
                        f"   • System Monitor:      http://localhost:{self.ports_cfg['system_monitor']}"
                    )
                    print(
                        f"   • Analytics Dashboard: http://localhost:{self.ports_cfg['analytics_dashboard']}"
                    )
                    print(
                        f"   • Performance Monitor: http://localhost:{self.ports_cfg['performance_monitor']}"
                    )

                    print("\n⚡ Complete Trading Architecture Active!")
                    print(
                        "   Mock/Live Feed → Signal Generator → Order Brain → Risk Manager → Dashboard"
                    )

                    # Open dashboards (unless headless)
                    self.open_dashboards()

                    print("\n💡 System is running. Press Ctrl+C to shutdown...")

                    # Start runtime monitoring
                    self.monitor_system_runtime()

                else:
                    print("\n⚠️  System health check failed. Some services may not be responding.")
                    print("   Please check the individual service logs for details.")

            except KeyboardInterrupt:
                pass
            finally:
                self.shutdown_all()


    def main():
        """Main entry point"""
        parser = argparse.ArgumentParser(description="Bambhoria Complete System Launcher")
        parser.add_argument(
            "--no-open", action="store_true", help="Do not open dashboards in browser"
        )
        parser.add_argument(
            "--no-watchdog",
            action="store_true",
            help="Disable watchdog auto-restart of crashed services",
        )
        args = parser.parse_args()

        launcher = BambhoriaSystemLauncher(no_open=args.no_open, no_watchdog=args.no_watchdog)
        launcher.launch_complete_system()


    if __name__ == "__main__":
        main()
        self.no_watchdog = no_watchdog
        self._running = True
        self._watchdog_thread = None

        # Ensure logs directory exists for process outputs
        self.logs_dir = Path("logs")
        try:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # Load ports/endpoints configuration with environment overrides
        self.ports_cfg = self._load_ports_config()

        # Define services. Only those with a 'port' are web dashboards we can probe.
        self.services = {
            "System Monitor": {
                "script": "system_monitor_dashboard.py",
                "port": self.ports_cfg["system_monitor"],
                "delay": 3,
            },
            "Analytics Dashboard": {
                "script": "dashboard_server.py",
                "port": self.ports_cfg["analytics_dashboard"],
                "delay": 3,
            },
            "Performance Monitor": {
                "script": "performance_dashboard.py",
                "port": self.ports_cfg["performance_monitor"],
                "delay": 2,
            },
            # http_mock_feed is a client that posts to an API; it does not expose an HTTP port
            "Mock Feed Client": {
                "script": "http_mock_feed.py",
                "port": None,
                "delay": 3,
                "args": ["--url", self.ports_cfg["mock_feed_api"]],
            },
            "Complete Trading System": {
                "script": "complete_trading_system.py",
                "port": None,
                "delay": 5,
            },
        }

        # Background monitor (no watchdog restarts for this one)
        self.background_services = {
            "Quantum Monitor": {"script": "quantum_performance_monitor.py", "port": None}
        }

    def _load_ports_config(self) -> dict:
        """Load ports/endpoints from config/ports.json with environment overrides."""
        cfg_path = Path("config/ports.json")
        data = {
            "system_monitor": 5008,
            "analytics_dashboard": 5000,
            "performance_monitor": 5009,
            "mock_feed_api": "http://localhost:5002/api/ticks",
        }
        try:
            if cfg_path.exists():
                with open(cfg_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        for k in data.keys():
                            if k in loaded:
                                data[k] = loaded[k]
        except Exception:
            # Fall back to defaults on any error
            pass

        # Environment overrides (if set)
        def _int_env(name: str, default: int) -> int:
            try:
                v = os.getenv(name)
                return int(v) if v is not None else default
            except Exception:
                return default

        data["system_monitor"] = _int_env("BAMBHORIA_PORT_SYSTEM_MONITOR", data["system_monitor"])
        data["analytics_dashboard"] = _int_env("BAMBHORIA_PORT_ANALYTICS", data["analytics_dashboard"])
        data["performance_monitor"] = _int_env("BAMBHORIA_PORT_PERF", data["performance_monitor"])
        api_env = os.getenv("BAMBHORIA_MOCK_FEED_API")
        if api_env:
            data["mock_feed_api"] = api_env

        return data

    def print_banner(self):
        banner = (
            "\n"
            "╔══════════════════════════════════════════════════════════════════════════════╗\n"
            "║                   🎯 BAMBHORIA COMPLETE TRADING SYSTEM 🎯                   ║\n"
            "║                         with Quantum Performance Monitoring                  ║\n"
            "╠══════════════════════════════════════════════════════════════════════════════╣\n"
            "║  🚀 AI-Powered Trading Pipeline                                             ║\n"
            "║  📊 Real-time Performance Monitoring                                        ║\n"
            "║  🛡️  Advanced Risk Management                                               ║\n"
            "║  💎 Multi-Dashboard Visualization                                           ║\n"
            "║  ⚡ Quantum System Health Monitoring                                        ║\n"
            "╚══════════════════════════════════════════════════════════════════════════════╝\n"
        )
        print(banner)

    def check_port_status(self, port: int, timeout: float = 3.0) -> bool:
        """Check if a port is open and responding on localhost."""
        try:
            response = requests.get(f"http://localhost:{port}", timeout=timeout)
            return response.status_code in (200, 404)  # 404 is acceptable for API endpoints
        except Exception:
            return False

    def start_service(self, name: str, config: dict):
        """Start a single service and stream its logs to a file."""
        script = config["script"]
        port = config.get("port")
        delay = config.get("delay", 2)
        extra_args = config.get("args", [])

        print(f"🚀 Starting {name}...")

        # Check if port is already serving something; if so, skip starting a server on it
        if port and self.check_port_status(port, timeout=1):
            print(f"⚠️  Port {port} already in use for {name}")
            return None

        try:
            # Stream output to log file to avoid PIPE buffer deadlocks
            log_path = self.logs_dir / f"{Path(script).stem}.log"
            log_file = open(log_path, "a", encoding="utf-8", buffering=1)
            argv = [sys.executable, script, *extra_args]
            process = subprocess.Popen(
                argv,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd(),
            )

            time.sleep(delay)

            # Check if process is still running
            if process.poll() is None:
                print(f"✅ {name} started successfully")
                return {
                    "name": name,
                    "process": process,
                    "port": port,
                    "config": config,
                    "restarts": 0,
                }
            else:
                print(f"❌ {name} failed to start")
                return None

        except Exception as e:
            print(f"❌ Error starting {name}: {e}")
            return None

    def start_background_monitor(self):
        """Start quantum performance monitor in background."""
        try:
            log_path = self.logs_dir / "quantum_performance_monitor.log"
            log_file = open(log_path, "a", encoding="utf-8", buffering=1)
            quantum_process = subprocess.Popen(
                [sys.executable, "quantum_performance_monitor.py"],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd(),
            )

            print("⚡ Quantum Performance Monitor running in background")
            return quantum_process
        except Exception as e:
            print(f"⚠️  Could not start Quantum Monitor: {e}")
            return None

    def open_dashboards(self):
        """Open all dashboards in a browser unless headless mode is enabled."""
        if self.no_open:
            print("\n🖥️  Headless mode (--no-open): skipping opening dashboards in browser")
            return

        print("\n🌐 Opening dashboards in browser...")
        dashboard_urls = [
            f"http://localhost:{self.ports_cfg['system_monitor']}",  # System Monitor
            f"http://localhost:{self.ports_cfg['analytics_dashboard']}",  # Analytics Dashboard
            f"http://localhost:{self.ports_cfg['performance_monitor']}",  # Performance Monitor
        ]

        for url in dashboard_urls:
            try:
                # Windows 'start' must be invoked via shell=True
                subprocess.run(["start", url], shell=True, check=False)
                time.sleep(1)
            except Exception:
                print(f"Could not open {url}")

    def _watchdog_loop(self):
        """Background watchdog: restarts failed services with exponential backoff."""
        MAX_RESTARTS = 5
        while self._running:
            try:
                for i, svc in enumerate(list(self.processes)):
                    name = svc.get("name")
                    if name == "Quantum Monitor":
                        # We do not restart the background monitor automatically
                        continue
                    proc = svc.get("process")
                    if not proc:
                        continue
                    # Process exited?
                    if proc.poll() is not None:
                        cfg = svc.get("config", {})
                        restarts = svc.get("restarts", 0)
                        if restarts >= MAX_RESTARTS:
                            print(f"❌ {name} exceeded max restarts; not restarting")
                            continue
                        backoff = min(2 ** restarts, 30)
                        print(
                            f"⚠️  {name} exited. Restarting in {backoff}s (attempt {restarts + 1}/{MAX_RESTARTS})"
                        )
                        time.sleep(backoff)
                        # Attempt restart
                        new_info = self.start_service(name, cfg)
                        if new_info:
                            new_info["restarts"] = restarts + 1
                            self.processes[i] = new_info
                            print(f"✅ {name} restarted")
                        else:
                            svc["restarts"] = restarts + 1
                            self.processes[i] = svc
                time.sleep(2)
            except Exception:
                time.sleep(2)

    def start_watchdog(self):
        if self.no_watchdog:
            print("🛡️  Watchdog disabled (--no-watchdog)")
            return
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            return
        print("🛡️  Starting watchdog thread")
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

    def health_check(self) -> bool:
        """Perform comprehensive health check.

        Returns True if overall health looks OK, False otherwise.
        """
        print("\n🔍 System Health Check:")
        print("-" * 50)

        # Check services that expose ports
        healthy_services = 0
        total_services = sum(1 for _n, cfg in self.services.items() if cfg.get("port"))

        for name, config in self.services.items():
            port = config.get("port")
            if port:
                if self.check_port_status(port):
                    print(f"✅ {name} (Port {port}): Healthy")
                    healthy_services += 1
                else:
                    print(f"❌ {name} (Port {port}): Not responding")

        # System resource check
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(".")

        print(f"\n💻 System Resources:")
        print(f"   CPU Usage: {cpu}%")
        print(f"   Memory Usage: {memory.percent}%")
        print(f"   Disk Usage: {round(disk.percent, 1)}%")
        print(f"   Available Memory: {round(memory.available / (1024**3), 2)} GB")

        # Overall health score
        service_health = (healthy_services / total_services) * 100 if total_services > 0 else 0
        resource_health = max(0, 100 - max(cpu - 50, memory.percent - 60, disk.percent - 70))
        overall_health = (service_health + resource_health) / 2

        print(f"\n🎯 Overall System Health: {overall_health:.1f}%")

        return overall_health > 70

    def monitor_system_runtime(self):
        """Monitor system during runtime."""
        print("\n📊 Real-time System Monitoring:")
        print("=" * 60)

        start_time = datetime.now()

        # Start watchdog in background
        self.start_watchdog()

        try:
            while True:
                current_time = datetime.now()
                uptime = current_time - start_time

                # Get current metrics
                cpu = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()

                # Check service health
                total_services = sum(1 for _n, cfg in self.services.items() if cfg.get("port"))
                active_services = 0
                for _name, config in self.services.items():
                    port = config.get("port")
                    if port and self.check_port_status(port, timeout=1):
                        active_services += 1

                # Display status
                status_line = (
                    f"[{current_time.strftime('%H:%M:%S')}] "
                    f"CPU: {cpu:5.1f}% | "
                    f"MEM: {memory.percent:5.1f}% | "
                    f"Services: {active_services}/{total_services} | "
                    f"Uptime: {str(uptime).split('.')[0]}"
                )

                print(f"\r{status_line}", end="", flush=True)

                time.sleep(5)

        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down system...")

    def shutdown_all(self):
        """Gracefully shutdown all services."""
        print("\n🛑 Shutting down all services...")

        self._running = False
        try:
            if self._watchdog_thread:
                self._watchdog_thread.join(timeout=1)
        except Exception:
            pass

        for service_info in self.processes:
            if service_info and service_info.get("process"):
                name = service_info.get("name", "(unknown)")
                process = service_info["process"]

                print(f"   Stopping {name}...")
                process.terminate()

                # Wait for graceful shutdown
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()

        print("✅ All services stopped successfully!")

    def launch_complete_system(self):
        """Launch the complete Bambhoria trading system."""
        self.print_banner()

        print(f"📂 Working Directory: {os.getcwd()}")
        print(f"🐍 Python: {sys.executable}")
        print(f"⏰ Launch Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            # Start all main services
            print("\n🚀 Starting Core Services:")
            print("=" * 40)

            for name, config in self.services.items():
                service_info = self.start_service(name, config)
                if service_info:
                    self.processes.append(service_info)

            # Start background quantum monitor
            quantum_process = self.start_background_monitor()
            if quantum_process:
                self.processes.append(
                    {"name": "Quantum Monitor", "process": quantum_process, "port": None}
                )

            # Wait for services to stabilize
            print("\n⏳ Waiting for services to initialize...")
            time.sleep(8)

            # Perform health check
            if self.health_check():
                print("\n🎉 System launched successfully!")

                # Show access information
                print(f"\n📊 Access Your Dashboards:")
                print(
                    f"   • System Monitor:      http://localhost:{self.ports_cfg['system_monitor']}"
                )
                print(
                    f"   • Analytics Dashboard: http://localhost:{self.ports_cfg['analytics_dashboard']}"
                )
                print(
                    f"   • Performance Monitor: http://localhost:{self.ports_cfg['performance_monitor']}"
                )

                print("\n⚡ Complete Trading Architecture Active!")
                print(
                    "   Mock/Live Feed → Signal Generator → Order Brain → Risk Manager → Dashboard"
                )

                # Open dashboards (unless headless)
                self.open_dashboards()

                print("\n💡 System is running. Press Ctrl+C to shutdown...")

                # Start runtime monitoring
                self.monitor_system_runtime()

            else:
                print("\n⚠️  System health check failed. Some services may not be responding.")
                print("   Please check the individual service logs for details.")

        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown_all()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Bambhoria Complete System Launcher")
    parser.add_argument(
        "--no-open", action="store_true", help="Do not open dashboards in browser"
    )
    parser.add_argument(
        "--no-watchdog",
        action="store_true",
        help="Disable watchdog auto-restart of crashed services",
    )
    args = parser.parse_args()

    launcher = BambhoriaSystemLauncher(no_open=args.no_open, no_watchdog=args.no_watchdog)
    launcher.launch_complete_system()


if __name__ == "__main__":
    main()
'''
# end legacy block
"""
complete_system_launcher.py
Bambhoria Complete System with Quantum Performance Monitoring
Ultimate launcher for the complete trading ecosystem
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil
import requests


class BambhoriaSystemLauncher:
    def __init__(self, no_open: bool = False, no_watchdog: bool = False):
        self.processes = []
        self.no_open = no_open
        self.no_watchdog = no_watchdog
        self._running = True
        self._watchdog_thread = None

        # Ensure logs directory exists for process outputs
        self.logs_dir = Path("logs")
        try:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # Load ports/endpoints configuration with environment overrides
        self.ports_cfg = self._load_ports_config()

        # Define services. Only those with a 'port' are web dashboards we can probe.
        self.services = {
            "System Monitor": {
                "script": "system_monitor_dashboard.py",
                "port": self.ports_cfg["system_monitor"],
                "delay": 3,
            },
            "Analytics Dashboard": {
                "script": "dashboard_server.py",
                "port": self.ports_cfg["analytics_dashboard"],
                "delay": 3,
            },
            "Performance Monitor": {
                "script": "performance_dashboard.py",
                "port": self.ports_cfg["performance_monitor"],
                "delay": 2,
            },
            # http_mock_feed is a client that posts to an API; it does not expose an HTTP port
            "Mock Feed Client": {
                "script": "http_mock_feed.py",
                "port": None,
                "delay": 3,
                "args": ["--url", self.ports_cfg["mock_feed_api"]],
            },
            "Complete Trading System": {
                "script": "complete_trading_system.py",
                "port": None,
                "delay": 5,
            },
        }

        # Background monitor (no watchdog restarts for this one)
        self.background_services = {
            "Quantum Monitor": {"script": "quantum_performance_monitor.py", "port": None}
        }

    def _load_ports_config(self) -> dict:
        """Load ports/endpoints from config/ports.json with environment overrides."""
        cfg_path = Path("config/ports.json")
        data = {
            "system_monitor": 5008,
            "analytics_dashboard": 5000,
            "performance_monitor": 5009,
            "mock_feed_api": "http://localhost:5002/api/ticks",
        }
        try:
            if cfg_path.exists():
                with open(cfg_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        for k in data.keys():
                            if k in loaded:
                                data[k] = loaded[k]
        except Exception:
            # Fall back to defaults on any error
            pass

        # Environment overrides (if set)
        def _int_env(name: str, default: int) -> int:
            try:
                v = os.getenv(name)
                return int(v) if v is not None else default
            except Exception:
                return default

        data["system_monitor"] = _int_env("BAMBHORIA_PORT_SYSTEM_MONITOR", data["system_monitor"])
        data["analytics_dashboard"] = _int_env("BAMBHORIA_PORT_ANALYTICS", data["analytics_dashboard"])
        data["performance_monitor"] = _int_env("BAMBHORIA_PORT_PERF", data["performance_monitor"])
        api_env = os.getenv("BAMBHORIA_MOCK_FEED_API")
        if api_env:
            data["mock_feed_api"] = api_env

        return data

    def print_banner(self):
        banner = (
            "\n"
            "╔══════════════════════════════════════════════════════════════════════════════╗\n"
            "║                   🎯 BAMBHORIA COMPLETE TRADING SYSTEM 🎯                   ║\n"
            "║                         with Quantum Performance Monitoring                  ║\n"
            "╠══════════════════════════════════════════════════════════════════════════════╣\n"
            "║  🚀 AI-Powered Trading Pipeline                                             ║\n"
            "║  📊 Real-time Performance Monitoring                                        ║\n"
            "║  🛡️  Advanced Risk Management                                               ║\n"
            "║  💎 Multi-Dashboard Visualization                                           ║\n"
            "║  ⚡ Quantum System Health Monitoring                                        ║\n"
            "╚══════════════════════════════════════════════════════════════════════════════╝\n"
        )
        print(banner)

    def check_port_status(self, port: int, timeout: float = 3.0) -> bool:
        """Check if a port is open and responding on localhost."""
        try:
            response = requests.get(f"http://localhost:{port}", timeout=timeout)
            return response.status_code in (200, 404)  # 404 is acceptable for API endpoints
        except Exception:
            return False

    def start_service(self, name: str, config: dict):
        """Start a single service and stream its logs to a file."""
        script = config["script"]
        port = config.get("port")
        delay = config.get("delay", 2)
        extra_args = config.get("args", [])

        print(f"🚀 Starting {name}...")

        # Check if port is already serving something; if so, skip starting a server on it
        if port and self.check_port_status(port, timeout=1):
            print(f"⚠️  Port {port} already in use for {name}")
            return None

        try:
            # Stream output to log file to avoid PIPE buffer deadlocks
            log_path = self.logs_dir / f"{Path(script).stem}.log"
            log_file = open(log_path, "a", encoding="utf-8", buffering=1)
            argv = [sys.executable, script, *extra_args]
            process = subprocess.Popen(
                argv,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd(),
            )

            time.sleep(delay)

            # Check if process is still running
            if process.poll() is None:
                print(f"✅ {name} started successfully")
                return {
                    "name": name,
                    "process": process,
                    "port": port,
                    "config": config,
                    "restarts": 0,
                }
            else:
                print(f"❌ {name} failed to start")
                return None

        except Exception as e:
            print(f"❌ Error starting {name}: {e}")
            return None

    def start_background_monitor(self):
        """Start quantum performance monitor in background."""
        try:
            log_path = self.logs_dir / "quantum_performance_monitor.log"
            log_file = open(log_path, "a", encoding="utf-8", buffering=1)
            quantum_process = subprocess.Popen(
                [sys.executable, "quantum_performance_monitor.py"],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd(),
            )

            print("⚡ Quantum Performance Monitor running in background")
            return quantum_process
        except Exception as e:
            print(f"⚠️  Could not start Quantum Monitor: {e}")
            return None

    def open_dashboards(self):
        """Open all dashboards in a browser unless headless mode is enabled."""
        if self.no_open:
            print("\n🖥️  Headless mode (--no-open): skipping opening dashboards in browser")
            return

        print("\n🌐 Opening dashboards in browser...")
        dashboard_urls = [
            f"http://localhost:{self.ports_cfg['system_monitor']}",  # System Monitor
            f"http://localhost:{self.ports_cfg['analytics_dashboard']}",  # Analytics Dashboard
            f"http://localhost:{self.ports_cfg['performance_monitor']}",  # Performance Monitor
        ]

        for url in dashboard_urls:
            try:
                # Windows 'start' must be invoked via shell=True
                subprocess.run(["start", url], shell=True, check=False)
                time.sleep(1)
            except Exception:
                print(f"Could not open {url}")

    def _watchdog_loop(self):
        """Background watchdog: restarts failed services with exponential backoff."""
        MAX_RESTARTS = 5
        while self._running:
            try:
                for i, svc in enumerate(list(self.processes)):
                    name = svc.get("name")
                    if name == "Quantum Monitor":
                        # We do not restart the background monitor automatically
                        continue
                    proc = svc.get("process")
                    if not proc:
                        continue
                    # Process exited?
                    if proc.poll() is not None:
                        cfg = svc.get("config", {})
                        restarts = svc.get("restarts", 0)
                        if restarts >= MAX_RESTARTS:
                            print(f"❌ {name} exceeded max restarts; not restarting")
                            continue
                        backoff = min(2 ** restarts, 30)
                        print(
                            f"⚠️  {name} exited. Restarting in {backoff}s (attempt {restarts + 1}/{MAX_RESTARTS})"
                        )
                        time.sleep(backoff)
                        # Attempt restart
                        new_info = self.start_service(name, cfg)
                        if new_info:
                            new_info["restarts"] = restarts + 1
                            self.processes[i] = new_info
                            print(f"✅ {name} restarted")
                        else:
                            svc["restarts"] = restarts + 1
                            self.processes[i] = svc
                time.sleep(2)
            except Exception:
                time.sleep(2)

    def start_watchdog(self):
        if self.no_watchdog:
            print("🛡️  Watchdog disabled (--no-watchdog)")
            return
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            return
        print("🛡️  Starting watchdog thread")
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

    def health_check(self) -> bool:
        """Perform comprehensive health check.

        Returns True if overall health looks OK, False otherwise.
        """
        print("\n🔍 System Health Check:")
        print("-" * 50)

        # Check services that expose ports
        healthy_services = 0
        total_services = sum(1 for _n, cfg in self.services.items() if cfg.get("port"))

        for name, config in self.services.items():
            port = config.get("port")
            if port:
                if self.check_port_status(port):
                    print(f"✅ {name} (Port {port}): Healthy")
                    healthy_services += 1
                else:
                    print(f"❌ {name} (Port {port}): Not responding")

        # System resource check
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(".")

        print(f"\n💻 System Resources:")
        print(f"   CPU Usage: {cpu}%")
        print(f"   Memory Usage: {memory.percent}%")
        print(f"   Disk Usage: {round(disk.percent, 1)}%")
        print(f"   Available Memory: {round(memory.available / (1024**3), 2)} GB")

        # Overall health score
        service_health = (healthy_services / total_services) * 100 if total_services > 0 else 0
        resource_health = max(0, 100 - max(cpu - 50, memory.percent - 60, disk.percent - 70))
        overall_health = (service_health + resource_health) / 2

        print(f"\n🎯 Overall System Health: {overall_health:.1f}%")

        return overall_health > 70

    def monitor_system_runtime(self):
        """Monitor system during runtime."""
        print("\n📊 Real-time System Monitoring:")
        print("=" * 60)

        start_time = datetime.now()

        # Start watchdog in background
        self.start_watchdog()

        try:
            while True:
                current_time = datetime.now()
                uptime = current_time - start_time

                # Get current metrics
                cpu = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()

                # Check service health
                total_services = sum(1 for _n, cfg in self.services.items() if cfg.get("port"))
                active_services = 0
                for _name, config in self.services.items():
                    port = config.get("port")
                    if port and self.check_port_status(port, timeout=1):
                        active_services += 1

                # Display status
                status_line = (
                    f"[{current_time.strftime('%H:%M:%S')}] "
                    f"CPU: {cpu:5.1f}% | "
                    f"MEM: {memory.percent:5.1f}% | "
                    f"Services: {active_services}/{total_services} | "
                    f"Uptime: {str(uptime).split('.')[0]}"
                )

                print(f"\r{status_line}", end="", flush=True)

                time.sleep(5)

        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down system...")

    def shutdown_all(self):
        """Gracefully shutdown all services."""
        print("\n🛑 Shutting down all services...")

        self._running = False
        try:
            if self._watchdog_thread:
                self._watchdog_thread.join(timeout=1)
        except Exception:
            pass

        for service_info in self.processes:
            if service_info and service_info.get("process"):
                name = service_info.get("name", "(unknown)")
                process = service_info["process"]

                print(f"   Stopping {name}...")
                process.terminate()

                # Wait for graceful shutdown
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()

        print("✅ All services stopped successfully!")

    def launch_complete_system(self):
        """Launch the complete Bambhoria trading system."""
        self.print_banner()

        print(f"📂 Working Directory: {os.getcwd()}")
        print(f"🐍 Python: {sys.executable}")
        print(f"⏰ Launch Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            # Start all main services
            print("\n🚀 Starting Core Services:")
            print("=" * 40)

            for name, config in self.services.items():
                service_info = self.start_service(name, config)
                if service_info:
                    self.processes.append(service_info)

            # Start background quantum monitor
            quantum_process = self.start_background_monitor()
            if quantum_process:
                self.processes.append(
                    {"name": "Quantum Monitor", "process": quantum_process, "port": None}
                )

            # Wait for services to stabilize
            print("\n⏳ Waiting for services to initialize...")
            time.sleep(8)

            # Perform health check
            if self.health_check():
                print("\n🎉 System launched successfully!")

                # Show access information
                print(f"\n📊 Access Your Dashboards:")
                print(
                    f"   • System Monitor:      http://localhost:{self.ports_cfg['system_monitor']}"
                )
                print(
                    f"   • Analytics Dashboard: http://localhost:{self.ports_cfg['analytics_dashboard']}"
                )
                print(
                    f"   • Performance Monitor: http://localhost:{self.ports_cfg['performance_monitor']}"
                )

                print("\n⚡ Complete Trading Architecture Active!")
                print(
                    "   Mock/Live Feed → Signal Generator → Order Brain → Risk Manager → Dashboard"
                )

                # Open dashboards (unless headless)
                self.open_dashboards()

                print("\n💡 System is running. Press Ctrl+C to shutdown...")

                # Start runtime monitoring
                self.monitor_system_runtime()

            else:
                print("\n⚠️  System health check failed. Some services may not be responding.")
                print("   Please check the individual service logs for details.")

        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown_all()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Bambhoria Complete System Launcher")
    parser.add_argument(
        "--no-open", action="store_true", help="Do not open dashboards in browser"
    )
    parser.add_argument(
        "--no-watchdog",
        action="store_true",
        help="Disable watchdog auto-restart of crashed services",
    )
    args = parser.parse_args()

    launcher = BambhoriaSystemLauncher(no_open=args.no_open, no_watchdog=args.no_watchdog)
    launcher.launch_complete_system()


if __name__ == "__main__":
    main()
"""
complete_system_launcher.py
Bambhoria Complete System with Quantum Performance Monitoring
Ultimate launcher for the complete trading ecosystem
"""

import subprocess
import time
import sys
import os
import requests
from threading import Thread
import psutil
from datetime import datetime
from pathlib import Path

class BambhoriaSystemLauncher:
    def __init__(self):
        self.processes = []
        # Ensure logs directory exists for process outputs
        self.logs_dir = Path("logs")
        try:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        # Load ports/endpoints configuration with environment overrides
        self.ports_cfg = self._load_ports_config()

        self.services = {
            "System Monitor": {"script": "system_monitor_dashboard.py", "port": self.ports_cfg["system_monitor"], "delay": 3},
            "Analytics Dashboard": {"script": "dashboard_server.py", "port": self.ports_cfg["analytics_dashboard"], "delay": 3},
            "Performance Monitor": {"script": "performance_dashboard.py", "port": self.ports_cfg["performance_monitor"], "delay": 2},
            # http_mock_feed is a client that posts to an API; it does not expose an HTTP port
            "Mock Feed Client": {"script": "http_mock_feed.py", "port": None, "delay": 3, "args": ["--url", self.ports_cfg["mock_feed_api"]]},
            "Complete Trading System": {"script": "complete_trading_system.py", "port": None, "delay": 5}
        }
        
        # Add quantum monitor as background process
        self.background_services = {
            "Quantum Monitor": {"script": "quantum_performance_monitor.py", "port": None}
        }
    
    def print_banner(self):
        banner = """
        """Load ports/endpoints from config/ports.json with environment overrides.
        Env variables:
          BAMBHORIA_PORT_SYSTEM_MONITOR, BAMBHORIA_PORT_ANALYTICS, BAMBHORIA_PORT_PERF,
          BAMBHORIA_MOCK_FEED_API
        """
        cfg_path = Path("config/ports.json")
        data = {
            "system_monitor": 5008,
            "analytics_dashboard": 5000,
            "performance_monitor": 5009,
            "mock_feed_api": "http://localhost:5002/api/ticks",
        }
        try:
            if cfg_path.exists():
                with open(cfg_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        data.update({k: loaded[k] for k in data.keys() if k in loaded})
        except Exception:
            pass
        # Environment overrides
        def _int_env(name, default):
            try:
                v = os.getenv(name)
                return int(v) if v is not None else default
            except Exception:
                return default
        data["system_monitor"] = _int_env("BAMBHORIA_PORT_SYSTEM_MONITOR", data["system_monitor"])
        data["analytics_dashboard"] = _int_env("BAMBHORIA_PORT_ANALYTICS", data["analytics_dashboard"])
        data["performance_monitor"] = _int_env("BAMBHORIA_PORT_PERF", data["performance_monitor"])
        api_env = os.getenv("BAMBHORIA_MOCK_FEED_API")
        if api_env:
            data["mock_feed_api"] = api_env
        return data
╔══════════════════════════════════════════════════════════════════════════════╗
║                   🎯 BAMBHORIA COMPLETE TRADING SYSTEM 🎯                   ║
║                         with Quantum Performance Monitoring                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🚀 AI-Powered Trading Pipeline                                             ║
║  📊 Real-time Performance Monitoring                                        ║
║  🛡️  Advanced Risk Management                                               ║
║  💎 Multi-Dashboard Visualization                                           ║
║  ⚡ Quantum System Health Monitoring                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def check_port_status(self, port, timeout=3):
        """Check if a port is open and responding"""
        try:
            response = requests.get(f"http://localhost:{port}", timeout=timeout)
            return response.status_code in [200, 404]  # 404 is acceptable for API endpoints
        except:
            return False
    
    def start_service(self, name, config):
        """Start a single service"""
        script = config["script"]
        port = config.get("port")
        delay = config.get("delay", 2)
    extra_args = config.get("args", [])
        
        print(f"🚀 Starting {name}...")
        
        # Check if port is already in use
        if port and self.check_port_status(port, timeout=1):
            print(f"⚠️  Port {port} already in use for {name}")
            return None
        
        try:
            # Stream output to log file to avoid PIPE buffer deadlocks
            log_path = self.logs_dir / f"{Path(script).stem}.log"
            log_file = open(log_path, "a", encoding="utf-8", buffering=1)
            process = subprocess.Popen(
                [sys.executable, script],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd(),
            )
            
            time.sleep(delay)
            
            # Check if process is still running
            if process.poll() is None:
                print(f"✅ {name} started successfully")
                return {"name": name, "process": process, "port": port}
            else:
                print(f"❌ {name} failed to start")
                return None
                
        except Exception as e:
            print(f"❌ Error starting {name}: {e}")
            return None
    
    def health_check(self):
        """Perform comprehensive health check"""
        print("\n🔍 System Health Check:")
        print("-" * 50)
        
        # Check services
        healthy_services = 0
        total_services = 0
        
        for name, config in self.services.items():
            port = config.get("port")
            if port:
                total_services += 1
                if self.check_port_status(port):
                    print(f"✅ {name} (Port {port}): Healthy")
                    healthy_services += 1
                else:
                    print(f"❌ {name} (Port {port}): Not responding")
        
        # System resource check
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
        
        print(f"\n💻 System Resources:")
        print(f"   CPU Usage: {cpu}%")
        print(f"   Memory Usage: {memory.percent}%")
        print(f"   Disk Usage: {round(disk.percent, 1)}%")
        print(f"   Available Memory: {round(memory.available / (1024**3), 2)} GB")
        
        # Overall health score
        service_health = (healthy_services / total_services) * 100 if total_services > 0 else 0
        resource_health = max(0, 100 - max(cpu-50, memory.percent-60, disk.percent-70))
        overall_health = (service_health + resource_health) / 2
        
        print(f"\n🎯 Overall System Health: {overall_health:.1f}%")
        
        return overall_health > 70
    
    def open_dashboards(self):
        """Open all dashboards in browser"""
        print("\n🌐 Opening dashboards in browser...")
        dashboard_urls = [
            f"http://localhost:{self.ports_cfg['system_monitor']}",  # System Monitor
            f"http://localhost:{self.ports_cfg['analytics_dashboard']}",  # Analytics Dashboard  
            f"http://localhost:{self.ports_cfg['performance_monitor']}",  # Performance Monitor
        ]
        
        for url in dashboard_urls:
            try:
                subprocess.run(["start", url], shell=True, check=False)
                time.sleep(1)
            except:
                print(f"Could not open {url}")
    
    def start_background_monitor(self):
        """Start quantum performance monitor in background"""
        try:
            log_path = self.logs_dir / "quantum_performance_monitor.log"
            log_file = open(log_path, "a", encoding="utf-8", buffering=1)
            quantum_process = subprocess.Popen(
                [sys.executable, "quantum_performance_monitor.py"],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd(),
            )
            
            print("⚡ Quantum Performance Monitor running in background")
            return quantum_process
        except Exception as e:
            print(f"⚠️  Could not start Quantum Monitor: {e}")
            return None
    
    def monitor_system_runtime(self):
        """Monitor system during runtime"""
        print("\n📊 Real-time System Monitoring:")
        print("=" * 60)
        
        start_time = datetime.now()
        
        try:
            while True:
                current_time = datetime.now()
                uptime = current_time - start_time
                
                # Get current metrics
                cpu = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # Check service health
                active_services = 0
                for name, config in self.services.items():
                    port = config.get("port")
                    if port and self.check_port_status(port, timeout=1):
                        active_services += 1
                
                # Display status
                status_line = (
                    f"[{current_time.strftime('%H:%M:%S')}] "
                    f"CPU: {cpu:5.1f}% | "
                    f"MEM: {memory.percent:5.1f}% | "
                    f"Services: {active_services}/4 | "
                    f"Uptime: {str(uptime).split('.')[0]}"
                )
                
                print(f"\r{status_line}", end="", flush=True)
                
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down system...")
    
    def shutdown_all(self):
        """Gracefully shutdown all services"""
        print("\n🛑 Shutting down all services...")
        
        for service_info in self.processes:
            if service_info and service_info["process"]:
                name = service_info["name"]
                process = service_info["process"]
                
                print(f"   Stopping {name}...")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        
        print("✅ All services stopped successfully!")
    
    def launch_complete_system(self):
        """Launch the complete Bambhoria trading system"""
        self.print_banner()
        
        print(f"📂 Working Directory: {os.getcwd()}")
        print(f"🐍 Python: {sys.executable}")
        print(f"⏰ Launch Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Start all main services
            print("\n🚀 Starting Core Services:")
            print("=" * 40)
            
            for name, config in self.services.items():
                service_info = self.start_service(name, config)
                if service_info:
                    self.processes.append(service_info)
            
            # Start background quantum monitor
            quantum_process = self.start_background_monitor()
            if quantum_process:
                self.processes.append({"name": "Quantum Monitor", "process": quantum_process, "port": None})
            
            # Wait for services to stabilize
            print("\n⏳ Waiting for services to initialize...")
            time.sleep(8)
            
            # Perform health check
            if self.health_check():
                print("\n🎉 System launched successfully!")
                
                # Show access information
                print(f"\n📊 Access Your Dashboards:")
                print(f"   • System Monitor:      http://localhost:5008")
                print(f"   • Analytics Dashboard: http://localhost:5000") 
                print(f"   • Performance Monitor: http://localhost:5009")
                print(f"   • System Monitor:      http://localhost:{self.ports_cfg['system_monitor']}")
                print(f"   • Analytics Dashboard: http://localhost:{self.ports_cfg['analytics_dashboard']}") 
                print(f"   • Performance Monitor: http://localhost:{self.ports_cfg['performance_monitor']}")
                
                print(f"\n⚡ Complete Trading Architecture Active!")
                print(f"   Mock/Live Feed → Signal Generator → Order Brain → Risk Manager → Dashboard")
                
                # Open dashboards
                self.open_dashboards()
                
                print(f"\n💡 System is running. Press Ctrl+C to shutdown...")
                
                # Start runtime monitoring
                self.monitor_system_runtime()
                
            else:
                print("\n⚠️  System health check failed. Some services may not be responding.")
                print("   Please check the individual service logs for details.")
                
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown_all()

def main():
    """Main entry point"""
    launcher = BambhoriaSystemLauncher()
    launcher.launch_complete_system()

if __name__ == "__main__":
    main()
'''
# end legacy block