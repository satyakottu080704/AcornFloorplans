#!/usr/bin/env python3
"""
Verify the pipeline feed — list files in the SharePoint Pending_Draw folder.
===========================================================================
Diagnostic: shows what (if anything) n8n has dropped for the renderer to pick
up. Relevant to the SharePoint-queue model; for the direct :8765 render-service
model the "feed" is the HTTP POST, not this folder.

Usage:
    python automation/list_pending_draw.py
    python automation/list_pending_draw.py --folder "General/AI Automation/Generated_Plans"

Reads SP_CLIENT_ID / SP_CLIENT_SECRET / SP_TENANT_ID / SP_DRIVE_ID from .env
(falls back to ACORN_* / OUTLOOK_* like the rest of the code).
"""
import os
import sys
import argparse
import urllib.parse
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass


def _graph_token() -> str:
    tenant = (os.getenv("SP_TENANT_ID") or os.getenv("ACORN_TENANT_ID") or os.getenv("OUTLOOK_TENANT_ID") or "").strip()
    cid = (os.getenv("SP_CLIENT_ID") or os.getenv("ACORN_CLIENT_ID") or os.getenv("OUTLOOK_CLIENT_ID") or "").strip()
    secret = (os.getenv("SP_CLIENT_SECRET") or os.getenv("ACORN_CLIENT_SECRET") or os.getenv("OUTLOOK_CLIENT_SECRET") or "").strip()
    if not (tenant and cid and secret):
        sys.exit("ERROR: SP_TENANT_ID / SP_CLIENT_ID / SP_CLIENT_SECRET not set in .env")
    r = requests.post(
        f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token",
        data={"client_id": cid, "client_secret": secret,
              "scope": "https://graph.microsoft.com/.default",
              "grant_type": "client_credentials"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def main():
    ap = argparse.ArgumentParser(description="List a SharePoint folder's files")
    ap.add_argument("--folder", default=os.getenv("SHAREPOINT_PENDING_FOLDER",
                    "General/AI Automation/Pending_Draw").strip("/"))
    args = ap.parse_args()

    drive = (os.getenv("SP_DRIVE_ID") or "").strip()
    if not drive:
        sys.exit("ERROR: SP_DRIVE_ID not set in .env")

    token = _graph_token()
    url = (f"https://graph.microsoft.com/v1.0/drives/{drive}/root:/"
           f"{urllib.parse.quote(args.folder)}:/children")
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=30)
    if r.status_code == 404:
        print(f"Folder not found / empty: {args.folder}")
        return
    r.raise_for_status()
    items = r.json().get("value", [])
    print(f"{args.folder}: {len(items)} item(s)")
    for it in sorted(items, key=lambda x: x.get("lastModifiedDateTime", "")):
        kb = (it.get("size", 0) or 0) / 1024
        print(f"  {kb:8.1f} KB  {it.get('lastModifiedDateTime','')[:19]}  {it['name']}")


if __name__ == "__main__":
    main()
