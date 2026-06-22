#!/usr/bin/env python3
"""
Acorn plan render service — runs on the Windows + Visio VM.
=========================================================
A tiny HTTP service the Hetzner pipeline/n8n calls directly:

    POST http://<windows-vm>:8765/render?project=N-12345
         body = the sketch image bytes (or multipart field "image")
         -> returns the generated .vsdx (COM/box pipeline, all fixes)

    GET  /health  -> {"ok": true}

Security: bind is 0.0.0.0:8765, but lock inbound to the Hetzner IP /32 in the
AWS security group + Windows Firewall. Optionally set RENDER_SERVICE_TOKEN and
send it as the `X-Auth-Token` header for defence-in-depth.

COM/Visio needs an INTERACTIVE desktop session — run this in a logged-in user
session (auto-logon + Task Scheduler "run only when user is logged on"), NOT a
Session-0 Windows service.

Run:  python automation/render_service.py     (uses waitress if installed)
"""
import io
import os
import sys
import tempfile
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

from flask import Flask, request, send_file, abort, jsonify

app = Flask(__name__)
# Surveyor sketch photos run large (we've seen up to ~9 MB) — allow generous
# request bodies so /render doesn't 413. Override with RENDER_MAX_MB if needed.
_max = int(os.getenv("RENDER_MAX_MB", "64")) * 1024 * 1024
app.config["MAX_CONTENT_LENGTH"] = _max
# Also raise the form-parse memory limit: if a client sends the body without a
# Content-Type (defaults to urlencoded), Flask tries to form-parse it and the
# default 500 KB form limit would 413 a big sketch. n8n's binaryData sends a
# proper type, but this makes /render robust either way.
app.config["MAX_FORM_MEMORY_SIZE"] = _max
TOKEN = os.getenv("RENDER_SERVICE_TOKEN", "").strip()
PORT = int(os.getenv("RENDER_PORT", "8765"))

# Deep-readiness state. /health says "Flask is alive"; /ready says "Visio COM
# actually launches" — set True only after a successful warmup so n8n waits for
# a VM that can really render, not just one whose web server is up.
_READY = {"visio": False, "error": "warming up"}

# Visio COM can only do ONE render at a time. Serialize all /render work behind
# this lock so overlapping requests (e.g. two n8n cycles overlapping) queue
# safely instead of colliding in COM. /health and /ready never take the lock,
# so readiness checks stay responsive even mid-render.
_RENDER_LOCK = threading.Lock()


def _warmup_visio():
    """Launch Visio COM once at boot: proves it works + pre-loads it so the
    first real render isn't a cold start. Runs in a daemon thread."""
    try:
        import pythoncom
        import win32com.client
        pythoncom.CoInitialize()
        try:
            app = win32com.client.Dispatch("Visio.Application")
            try:
                app.Visible = False
            except Exception:
                pass
            try:
                app.Quit()
            except Exception:
                pass
            _READY["visio"] = True
            _READY["error"] = None
            print("[render-service] Visio COM warmup OK — ready to render")
        finally:
            pythoncom.CoUninitialize()
    except Exception as e:  # COM/Visio not available yet (or at all)
        _READY["error"] = str(e)
        print(f"[render-service] Visio COM warmup FAILED: {e}")


@app.get("/health")
def health():
    # Liveness only — the web server is up. Does NOT imply Visio is ready.
    return jsonify(ok=True, service="acorn-render", visio="com")


@app.get("/ready")
def ready():
    # Readiness — 200 only once Visio COM has been confirmed working.
    if _READY["visio"]:
        return jsonify(ok=True, ready=True, visio="com")
    return jsonify(ok=False, ready=False, error=_READY["error"]), 503


@app.post("/render")
def render():
    # Optional shared-secret check (in addition to the IP allow-list).
    # Accept either header name so n8n's X-Agent-Token or X-Auth-Token works.
    if TOKEN:
        sent = (request.headers.get("X-Auth-Token")
                or request.headers.get("X-Agent-Token") or "")
        if sent != TOKEN:
            abort(401)

    project = (request.args.get("project")
               or request.headers.get("X-Project")
               or "N-99999").strip()
    # Sanitise: if the caller sent an unevaluated n8n expression ("{{ ... }}")
    # or anything with path/illegal chars, keep only safe filename characters so
    # temp files are named sanely and can't collide/pile up.
    safe = "".join(ch for ch in project if ch.isalnum() or ch in "._-")
    project = safe or "N-99999"

    # Accept either a multipart "image" field or raw image bytes in the body.
    if "image" in request.files:
        data = request.files["image"].read()
    else:
        data = request.get_data()
    if not data:
        abort(400, "no image supplied")

    tmp = Path(tempfile.gettempdir())
    in_path = tmp / f"{project}_sketch.jpg"
    out_path = tmp / f"{project}.vsdx"
    in_path.write_bytes(data)
    _RENDER_LOCK.acquire()  # serialize: Visio COM can only render one at a time
    try:
        from pipeline import process_sketch  # imported here so /health works without Visio
        vsdx, plan = process_sketch(str(in_path), output_path=str(out_path))
        if not (vsdx and os.path.exists(vsdx)):
            abort(500, "render produced no vsdx")
        # Read the result into memory so we can delete the temp file immediately
        # (in the finally block) — the bytes are safely in the response, so the
        # VM temp folder never accumulates rendered .vsdx files.
        vsdx_bytes = Path(vsdx).read_bytes()
        rooms, samples = len(plan.rooms), len(plan.samples)
        _READY["visio"] = True  # a real render just proved Visio works
        _READY["error"] = None
        resp = send_file(io.BytesIO(vsdx_bytes), as_attachment=True,
                         download_name=f"{project} Floor Plan.vsdx",
                         mimetype="application/vnd.visio")
        resp.headers["X-Rooms"] = str(rooms)
        resp.headers["X-Samples"] = str(samples)
        return resp
    except Exception as e:
        abort(500, f"render failed: {e}")
    finally:
        _RENDER_LOCK.release()  # let the next queued render proceed
        # Always clean up BOTH temp files (input sketch + output vsdx) so the
        # VM temp folder doesn't fill up over a high-volume week.
        for _p in (in_path, out_path):
            try:
                _p.unlink()
            except OSError:
                pass


def main():
    print(f"[render-service] listening on 0.0.0.0:{PORT}  (token={'set' if TOKEN else 'OFF'})")
    # Warm Visio COM in the background so /ready flips to 200 once it's confirmed
    # working — without blocking the web server from binding the port.
    threading.Thread(target=_warmup_visio, name="visio-warmup", daemon=True).start()
    try:
        from waitress import serve  # production WSGI server if available
        serve(app, host="0.0.0.0", port=PORT, threads=2)
    except ImportError:
        app.run(host="0.0.0.0", port=PORT, threaded=False)  # dev fallback (1 render at a time)


if __name__ == "__main__":
    main()
