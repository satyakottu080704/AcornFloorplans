#!/usr/bin/env python3
"""
Wates plan orchestrator — runs INSIDE the acorn_reporting container.
===================================================================

This is the container-native entrypoint n8n drives over SSH. It reuses the
container's OWN code (the `api` package + `src/plans/generate_plan.py`, which
draws via Gemini -> SVG -> Aspose -> VSDX). It does NOT use the
AcornPlanGeneration `pipeline`/`process_sketch` (not present in this container).

Steps:
  1. GET project from AlphaTracker (api.get_project).
  2. Gate: clientName contains "wates" AND status == "Scheduled".
  3. Draw the plan by calling src/plans/generate_plan.py -> "<N> AI Draft.vsdx".
  4. Upload that VSDX to the configured SharePoint review or production folder.

AlphaTracker file upload and status updates are intentionally disabled.

n8n call:
  docker exec acorn_reporting python src/process_plan.py "N-12345" --image /tmp/<file>

Prints ONE JSON line. Exit 0 = handled OR skipped (n8n marks email read);
exit 2 = transient failure (n8n leaves it unread to retry).
"""
import os
import sys
import json
import argparse
import subprocess

SRC = "/app/src"
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from api import get_project, upload_file, update_project  # container's api package

import requests
import urllib.parse

OUT_DIR = os.environ.get("PLANS_OUTPUT_DIR", "/app/src/output/reports")
FILE_TYPE = os.environ.get("YELLOW_FOLDER_FILE_TYPE", "survey_plan_visio")
STATUS = os.environ.get("PLANS_DRAWN_STATUS", "Plans Drawn")
REQUIRE_WATES_CLIENT = os.environ.get(
    "PLAN_REQUIRE_WATES_CLIENT", "false"
).strip().lower() in ("true", "1", "yes")
DEFAULT_REVIEW_FOLDER = "General/AI Automation/Manual_Review"
DEFAULT_PRODUCTION_FOLDER = "General/AI Automation/Generated_Plans"


def _acquire_graph_token():
    """Acquire Microsoft Graph access token using credentials from environment."""
    tenant = (os.environ.get("SP_TENANT_ID") or os.environ.get("ACORN_TENANT_ID") or os.environ.get("OUTLOOK_TENANT_ID") or "").strip()
    client_id = (os.environ.get("SP_CLIENT_ID") or os.environ.get("ACORN_CLIENT_ID") or os.environ.get("OUTLOOK_CLIENT_ID") or "").strip()
    secret = (os.environ.get("SP_CLIENT_SECRET") or os.environ.get("ACORN_CLIENT_SECRET") or os.environ.get("OUTLOOK_CLIENT_SECRET") or "").strip()
    
    if not (tenant and client_id and secret):
        return None
        
    url = f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
    data = {
        "client_id": client_id,
        "client_secret": secret,
        "scope": "https://graph.microsoft.com/.default",
        "grant_type": "client_credentials",
    }
    try:
        r = requests.post(url, data=data, timeout=30)
        r.raise_for_status()
        return r.json().get("access_token")
    except Exception:
        return None


def _publish_target():
    """Return the explicit publish mode and corresponding SharePoint folder.

    'auto' defers the folder to the per-plan quality gate (pass -> production
    folder, fail -> review folder), so structurally-sound plans publish with no
    human step and only questionable ones stop for review.
    """
    mode = (os.environ.get("PLAN_PUBLISH_MODE") or "review").strip().lower()
    if mode == "review":
        folder = os.environ.get("SHAREPOINT_REVIEW_FOLDER") or DEFAULT_REVIEW_FOLDER
    elif mode == "production":
        folder = os.environ.get("SHAREPOINT_OUTPUT_FOLDER") or DEFAULT_PRODUCTION_FOLDER
    elif mode == "auto":
        return mode, None  # resolved per-plan by the quality gate in run()
    else:
        raise ValueError("PLAN_PUBLISH_MODE must be 'review', 'production', or 'auto'")
    return mode, folder.strip().strip("/")


def _load_acceptance_report(path):
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception as exc:
        return {"_error": str(exc)}


def _project_accepted_by_report(project_number, report):
    """Return (accepted, reason) for production publishing."""
    if not isinstance(report, dict):
        return False, "acceptance report missing"
    if report.get("_error"):
        return False, f"acceptance report unreadable: {report['_error']}"

    pn = str(project_number or "").strip().lower()
    for row in report.get("results") or []:
        row_pn = str(row.get("project_number") or "").strip().lower()
        row_key = str(row.get("key") or "").strip().lower()
        if pn not in {row_pn, row_key}:
            continue
        if row.get("status") != "ok":
            return False, f"acceptance status {row.get('status')}"
        if int(row.get("accepted") or 0) != 1:
            return False, f"acceptance thresholds not met: {row.get('notes') or 'thresholds_not_met'}"
        return True, None
    return False, "project not found in acceptance report"


def _production_acceptance_reason(project_number):
    report_path = os.environ.get("PLAN_ACCEPTANCE_REPORT", "").strip()
    if not report_path:
        return "PLAN_ACCEPTANCE_REPORT is required for production mode"
    accepted, reason = _project_accepted_by_report(
        project_number,
        _load_acceptance_report(report_path),
    )
    return None if accepted else reason


def _ensure_sharepoint_folder(token, drive_id, relative_folder):
    """Create missing SharePoint path segments without replacing existing folders."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    parent = ""
    for segment in relative_folder.split("/"):
        if not segment:
            continue
        if parent:
            escaped_parent = urllib.parse.quote(parent)
            url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root:/{escaped_parent}:/children"
        else:
            url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root/children"
        response = requests.post(
            url,
            headers=headers,
            json={"name": segment, "folder": {}, "@microsoft.graph.conflictBehavior": "fail"},
            timeout=30,
        )
        if response.status_code not in (200, 201, 409):
            raise RuntimeError(
                f"Unable to ensure SharePoint folder '{parent}/{segment}': "
                f"HTTP {response.status_code}: {response.text}"
            )
        parent = f"{parent}/{segment}".strip("/")


def upload_vsdx_to_sharepoint(pn, file_path, relative_folder=None) -> dict:
    """Upload generated VSDX to the SharePoint document library."""
    token = _acquire_graph_token()
    if not token:
        print("ERROR: Failed to acquire MS Graph token for SharePoint upload", file=sys.stderr)
        return {"success": False, "error": "Failed to acquire MS Graph token"}
        
    filename = os.path.basename(file_path)
    drive_id = (os.environ.get("SP_DRIVE_ID") or "").strip()
    if not drive_id:
        return {"success": False, "error": "SP_DRIVE_ID is not configured"}
    relative_folder = (
        relative_folder
        or os.environ.get("SHAREPOINT_OUTPUT_FOLDER")
        or DEFAULT_PRODUCTION_FOLDER
    ).strip().strip("/")

    try:
        _ensure_sharepoint_folder(token, drive_id, relative_folder)
    except Exception as exc:
        return {"success": False, "error": str(exc)}
    
    # Target URL via specific Drive ID
    escaped_folder = urllib.parse.quote(relative_folder)
    escaped_filename = urllib.parse.quote(filename)
    url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root:/{escaped_folder}/{escaped_filename}:/content"
    
    print(f"INFO: Uploading {filename} to SharePoint: drive={drive_id}, folder={relative_folder}, url={url}", file=sys.stderr)
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/octet-stream"
    }
    
    try:
        if not os.path.exists(file_path):
            print(f"ERROR: SharePoint upload failed: File not found at {file_path}", file=sys.stderr)
            return {"success": False, "error": f"File not found: {file_path}"}
            
        with open(file_path, "rb") as f:
            data = f.read()
            
        r = requests.put(url, headers=headers, data=data, timeout=60)
        if r.status_code in (200, 201):
            web_url = r.json().get("webUrl")
            print(f"INFO: Successfully uploaded {filename} to SharePoint: {web_url}", file=sys.stderr)
            return {
                "success": True, 
                "web_url": web_url, 
                "sharepointPath": f"drives/{drive_id}/root:/{relative_folder}/{filename}"
            }
        else:
            print(f"ERROR: SharePoint upload failed with status {r.status_code}: {r.text}", file=sys.stderr)
            return {"success": False, "error": f"HTTP {r.status_code}: {r.text}"}
    except Exception as e:
        print(f"ERROR: SharePoint upload exception: {e}", file=sys.stderr)
        return {"success": False, "error": str(e)}


def upload_to_trackerfiler(pn: str, file_path: str) -> dict:
    """Send the generated Visio plan to trackerfiler email so AT files it under the project."""
    token = _acquire_graph_token()
    if not token:
        return {"success": False, "error": "No MS Graph token acquired"}

    import base64
    filename = os.path.basename(file_path)
    
    try:
        if not os.path.exists(file_path):
            return {"success": False, "error": f"File not found: {file_path}"}
            
        with open(file_path, "rb") as f:
            file_b64 = base64.b64encode(f.read()).decode()
            
        mail_from = (os.environ.get("MAIL_FROM_USER") or os.environ.get("ACORN_USER_PRINCIPAL_NAME") or "software@acornasbestos.co.uk").strip()
        mail_tracker = (os.environ.get("MAIL_TRACKERFILER") or "trackerfiler@acornasbestos.co.uk").strip()
        
        url = f"https://graph.microsoft.com/v1.0/users/{mail_from}/sendMail"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "message": {
                "subject": f"{pn}",
                "body": {"contentType": "Text", "content": f"AI generated Visio plan draft for {pn}"},
                "toRecipients": [{"emailAddress": {"address": mail_tracker}}],
                "attachments": [{
                    "@odata.type": "#microsoft.graph.fileAttachment",
                    "name": filename,
                    "contentType": "application/vnd.visio",
                    "contentBytes": file_b64
                }]
            },
            "saveToSentItems": False
        }
        
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        if r.status_code in (200, 202):
            return {"success": True}
        else:
            return {"success": False, "error": f"HTTP {r.status_code}: {r.text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _field(project, key):
    val = project.get(key)
    if not val and isinstance(project.get("data"), dict):
        val = project["data"].get(key)
    return str(val or "")


def _gate(project):
    if REQUIRE_WATES_CLIENT and "wates" not in _field(project, "clientName").lower():
        return "not wates"
    if _field(project, "status").strip().lower() != "scheduled":
        return "not scheduled"
    return None


def _ok(resp):
    """Tolerant success check across possible api return shapes."""
    if not isinstance(resp, dict):
        return bool(resp)
    if resp.get("success") is True:
        return True
    if str(resp.get("status", "")).lower() in ("success", "ok"):
        return True
    return "error" not in resp and "success" not in resp and bool(resp)


# Quality-gate thresholds (tunable via env without redeploying).
MIN_ROOMS = int(os.environ.get("PLAN_MIN_ROOMS", "1") or "1")
MIN_WALLS = int(os.environ.get("PLAN_MIN_WALLS", "4") or "4")


def _load_quality(vsdx_path):
    """Load the quality sidecar generate_plan.py wrote next to the VSDX."""
    side = os.path.splitext(vsdx_path)[0] + ".quality.json"
    try:
        with open(side, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return None


def _quality_gate(quality):
    """Self-contained pre-publish gate (NO ground truth). Returns (passed, reasons).

    Catches GROSS structural failures so they never auto-publish: placeholder
    layouts, zero/too-few rooms, unlabeled rooms, no enclosing walls,
    out-of-bounds coordinates, unlabeled samples. It does NOT prove the plan
    matches the sketch — a structurally-valid plan can still be factually wrong
    (that requires the eval harness / a human).
    """
    if not isinstance(quality, dict):
        return False, ["no quality summary produced"]
    reasons = []
    if not quality.get("layout_is_real"):
        reasons.append("placeholder/dummy layout")
    if int(quality.get("room_count") or 0) < MIN_ROOMS:
        reasons.append(f"too few rooms (<{MIN_ROOMS})")
    if int(quality.get("blank_label_count") or 0) > 0:
        reasons.append(f"{quality.get('blank_label_count')} unlabeled room(s)")
    if int(quality.get("wall_count") or 0) < MIN_WALLS:
        reasons.append(f"too few walls (<{MIN_WALLS}) — not an enclosed plan")
    if not quality.get("coords_in_bounds"):
        reasons.append("coordinates out of bounds")
    if int(quality.get("blank_sample_count") or 0) > 0:
        reasons.append("unlabeled sample marker(s)")
    return (len(reasons) == 0), reasons


def run(pn, image, dry_run):
    try:
        publish_mode, publish_folder = _publish_target()
    except ValueError as exc:
        return {"error": str(exc), "projectNumber": pn}, 2

    project = get_project(pn)
    if not project:
        return {"error": "project not found / AT call failed", "projectNumber": pn}, 2

    reason = _gate(project)
    if reason:
        return {"skipped": True, "reason": reason, "projectNumber": pn}, 0

    if not image or not os.path.exists(image):
        return {"error": f"image not found: {image}", "projectNumber": pn}, 2

    vsdx = os.path.join(OUT_DIR, f"{pn} AI Draft.vsdx")
    try:
        if os.path.exists(vsdx):
            os.remove(vsdx)
    except OSError:
        pass

    # Aspose.Diagram (.NET) needs ICU or invariant globalization; the container
    # has no compatible ICU, so run generation in invariant mode (no ICU needed).
    gen_env = {**os.environ, "DOTNET_SYSTEM_GLOBALIZATION_INVARIANT": "1"}
    proc = subprocess.run(
        [sys.executable, "src/plans/generate_plan.py", pn, "--image", image],
        cwd="/app", env=gen_env, capture_output=True, text=True, timeout=540,
    )
    if not os.path.exists(vsdx):
        tail = (proc.stdout[-400:] + proc.stderr[-400:]).strip()
        return {"error": "generation produced no vsdx", "projectNumber": pn, "detail": tail}, 2

    # Self-contained quality gate (NO ground truth): structural sanity of the
    # generated plan. Pass = safe to auto-publish; fail = route to human review.
    quality = _load_quality(vsdx)
    gate_passed, gate_reasons = _quality_gate(quality)

    if dry_run:
        return {
            "ok": True,
            "dryRun": True,
            "projectNumber": pn,
            "vsdx": vsdx,
            "publishMode": publish_mode,
            "qualityGatePassed": gate_passed,
            "qualityGateReasons": gate_reasons,
            "quality": quality,
            "reviewRequired": (publish_mode == "review") or not gate_passed,
            "targetFolder": publish_folder,
        }, 0

    # 'auto' mode: the gate picks the folder — a clean plan publishes straight to
    # the production folder, a questionable one is routed to human review.
    if publish_mode == "auto":
        publish_folder = (
            (os.environ.get("SHAREPOINT_OUTPUT_FOLDER") or DEFAULT_PRODUCTION_FOLDER)
            if gate_passed
            else (os.environ.get("SHAREPOINT_REVIEW_FOLDER") or DEFAULT_REVIEW_FOLDER)
        ).strip().strip("/")

    if publish_mode == "production":
        acceptance_reason = _production_acceptance_reason(pn)
        if acceptance_reason:
            return {
                "error": "production acceptance failed",
                "reason": acceptance_reason,
                "qualityGateReasons": gate_reasons,
                "projectNumber": pn,
                "generated": True,
                "publishMode": publish_mode,
                "reviewRequired": True,
                "suggestedTargetFolder": DEFAULT_REVIEW_FOLDER,
            }, 2

    # DISABLE saving to Alpha Tracker for now (file upload and status updates)
    # up = upload_file(pn, vsdx, FILE_TYPE)
    # if not _ok(up):
    #     return {"error": "upload failed", "detail": up, "projectNumber": pn,
    #             "generated": True, "uploaded": False}, 2
    #
    # st = update_project(pn, {"status": STATUS})
    # if not _ok(st):
    #     return {"error": "status update failed", "detail": st, "projectNumber": pn,
    #             "generated": True, "uploaded": True, "statusSet": False}, 2

    # Native package/visibility validation happens in generate_plan.py. Until
    # geometry acceptance is signed off, the default mode queues drafts for
    # human review instead of publishing them as completed plans.
    sp_res = upload_vsdx_to_sharepoint(pn, vsdx, publish_folder)
    if not sp_res.get("success"):
        return {"error": "SharePoint upload failed", "detail": sp_res.get("error"), "projectNumber": pn,
                "generated": True, "uploaded": False, "sharepointUploaded": False}, 2

    # Optional Trackerfiler upload (disabled by default)
    tf_uploaded = False
    tf_error = None
    if (
        publish_mode == "production"
        and os.getenv("UPLOAD_TO_TRACKERFILER", "false").strip().lower() in ("true", "1", "yes")
    ):
        tf_res = upload_to_trackerfiler(pn, vsdx)
        tf_uploaded = tf_res.get("success", False)
        tf_error = tf_res.get("error") if not tf_uploaded else None

    # Clean up local sketch and VSDX files to save disk space
    try:
        if image and os.path.exists(image):
            os.remove(image)
        if vsdx and os.path.exists(vsdx):
            os.remove(vsdx)
    except Exception:
        pass

    return {"ok": True, "projectNumber": pn, "generated": True,
            "uploaded": False, "statusSet": False, "sharepointUploaded": True,
            "sharepointUrl": sp_res.get("web_url"),
            "sharepointPath": sp_res.get("sharepointPath"),
            "publishMode": publish_mode,
            "qualityGatePassed": gate_passed,
            "qualityGateReasons": gate_reasons,
            "autoPublished": publish_mode == "auto" and gate_passed,
            "reviewRequired": (publish_mode == "review") or not gate_passed,
            "targetFolder": publish_folder,
            "trackerfilerSent": tf_uploaded,
            "trackerfilerError": tf_error}, 0


def main():
    ap = argparse.ArgumentParser(description="Wates plan orchestrator (acorn_reporting container)")
    ap.add_argument("project_number")
    ap.add_argument("--image", required=True)
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="Gate + draw, but skip the AT upload + status writes")
    args = ap.parse_args()
    result, code = run(args.project_number, args.image, args.dry_run)
    print(json.dumps(result))
    sys.exit(code)


if __name__ == "__main__":
    main()
