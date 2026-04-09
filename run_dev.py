#!/usr/bin/env python
"""Development server with auto-reload for Swastha Sevak.

This script uses watchdog to monitor file changes and restart the server.
It avoids uvicorn's reload mechanism which has issues on Windows with package imports.
"""

import subprocess
import sys
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class ChangeHandler(FileSystemEventHandler):
    def __init__(self, process_controller):
        self.process_controller = process_controller

    def on_modified(self, event):
        if not event.is_directory:
            # Only watch Python files in app directory
            path = Path(event.src_path)
            if path.suffix == ".py" and "app" in str(path):
                print(f"\n[CHANGE DETECTED] {path.relative_to(Path.cwd())}")
                self.process_controller.restart()

    def on_created(self, event):
        if not event.is_directory:
            path = Path(event.src_path)
            if path.suffix == ".py" and "app" in str(path):
                print(f"\n[FILE CREATED] {path.relative_to(Path.cwd())}")
                self.process_controller.restart()


class ProcessController:
    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        self.host = host
        self.port = port
        self.process = None

    def start(self):
        print(f"Starting Swastha Sevak on {self.host}:{self.port}...")
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                self.host,
                "--port",
                str(self.port),
            ],
            cwd=Path(__file__).parent,
        )

    def restart(self):
        print("\n[RESTARTING] Server...")
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        time.sleep(1)
        self.start()

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Swastha Sevak dev server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--no-watch",
        action="store_true",
        help="Run without file watching (no auto-reload)",
    )
    args = parser.parse_args()

    controller = ProcessController(host=args.host, port=args.port)
    controller.start()

    if not args.no_watch:
        observer = Observer()
        handler = ChangeHandler(controller)
        observer.schedule(handler, path="app", recursive=True)
        observer.start()

        try:
            print("\n[WATCHING] for changes in ./app/ (Ctrl+C to stop)")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
            observer.join()
            controller.stop()
            print("\n[STOPPED] Swastha Sevak")
    else:
        try:
            controller.process.wait()
        except KeyboardInterrupt:
            controller.stop()
            print("\n[STOPPED] Swastha Sevak")
