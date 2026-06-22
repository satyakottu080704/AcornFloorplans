#!/usr/bin/env python3
"""
Fetch plan-sketch images from the Plans inbox into input/ for local testing.
============================================================================
Reads recent messages from the Plans mailbox via Microsoft Graph (app-only),
grabs the largest image attachment from each email that has an N-number in the
subject, and saves it as input/<N-number>.jpg. Used to build a local test set
for tuning geometry (main.py).

    python automation/fetch_inbox_sketches.py [N]      # N = how many recent emails to scan (default 60)

Creds from .env: ACORN_* (falls back to OUTLOOK_*); mailbox = PLANS_MAILBOX or
ACORN_USER_PRINCIPAL_NAME.
"""
import os
import re
import sys
import base64
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def env(*names, default=""):
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return default


TENANT = env("ACORN_TENANT_ID", "OUTLOOK_TENANT_ID")
CID = env("ACORN_CLIENT_ID", "OUTLOOK_CLIENT_ID")
SECRET = env("ACORN_CLIENT_SECRET", "OUTLOOK_CLIENT_SECRET")
MAILBOX = env("PLANS_MAILBOX", "ACORN_USER_PRINCIPAL_NAME", "OUTLOOK_EMAIL")
OUT = ROOT / "input"
OUT.mkdir(exist_ok=True)
SCAN = int(sys.argv[1]) if len(sys.argv) > 1 else 60

if not (TENANT and CID and SECRET and MAILBOX):
    sys.exit("Missing creds/mailbox in .env (ACORN_TENANT_ID/CLIENT_ID/CLIENT_SECRET + ACORN_USER_PRINCIPAL_NAME)")

tok = requests.post(
    f"https://login.microsoftonline.com/{TENANT}/oauth2/v2.0/token",
    data={"client_id": CID, "client_secret": SECRET,
          "scope": "https://graph.microsoft.com/.default",
          "grant_type": "client_credentials"},
    timeout=30,
).json()
if "access_token" not in tok:
    sys.exit(f"Token failed: {tok}")
H = {"Authorization": f"Bearer {tok['access_token']}"}

url = (f"https://graph.microsoft.com/v1.0/users/{MAILBOX}/mailFolders/inbox/messages"
       f"?$top={SCAN}&$select=id,subject,hasAttachments,receivedDateTime"
       f"&$orderby=receivedDateTime desc")
r = requests.get(url, headers=H, timeout=30)
if r.status_code != 200:
    sys.exit(f"Cannot read inbox of {MAILBOX}: HTTP {r.status_code} {r.text[:300]}")
msgs = r.json().get("value", [])
print(f"Mailbox {MAILBOX}: scanned {len(msgs)} recent messages")

saved = 0
seen = set()
for m in msgs:
    if not m.get("hasAttachments"):
        continue
    subj = m.get("subject", "") or ""
    nm = re.search(r"N-\d{4,10}", subj, re.I)
    pn = nm.group(0).upper() if nm else None
    if pn and pn in seen:
        continue
    atts = requests.get(
        f"https://graph.microsoft.com/v1.0/users/{MAILBOX}/messages/{m['id']}/attachments",
        headers=H, timeout=30,
    ).json().get("value", [])
    imgs = [a for a in atts
            if a.get("@odata.type", "").endswith("fileAttachment")
            and (str(a.get("contentType", "")).startswith("image/")
                 or re.search(r"\.(jpe?g|png)$", a.get("name", ""), re.I))]
    if not imgs:
        continue
    big = max(imgs, key=lambda a: a.get("size", 0))
    base = pn or (re.sub(r"[^A-Za-z0-9._-]", "_", subj)[:40] or m["id"][:12])
    ext = (big.get("name", "x.jpg").rsplit(".", 1)[-1] or "jpg").lower()
    if ext not in ("jpg", "jpeg", "png"):
        ext = "jpg"
    data = base64.b64decode(big["contentBytes"])
    (OUT / f"{base}.{ext}").write_bytes(data)
    if pn:
        seen.add(pn)
    saved += 1
    print(f"  saved {base}.{ext:4}  {len(data):>8} B   <- {subj[:55]}")

print(f"\nDone: {saved} sketches saved to {OUT}")
