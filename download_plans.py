"""Download floor plan images from OneDrive into data/images/.

Talks to Microsoft Graph using a personal access token from Graph Explorer
(https://developer.microsoft.com/en-us/graph/graph-explorer). Walks
/Desktop/Documents/PLANS/ recursively and pulls every .jpg/.jpeg/.png it
finds. Skips files already present locally.

First run:
    1. Open Graph Explorer, sign in with the Acorn account that owns the
       PLANS folder.
    2. Consent to the "Files.Read" permission (Modify permissions tab).
    3. Run any sample query so the token is minted.
    4. Click your avatar -> "Access token" -> copy.
    5. Paste it when this script prompts, or put it in .env as
       MSGRAPH_TOKEN=<token>. Tokens last ~1 hour; re-paste when expired.

Usage:
    python download_plans.py              # download everything new
    python download_plans.py --limit 10   # smoke test
    python download_plans.py --dry-run    # list what would be downloaded
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterator, Optional

import requests

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"
OUTPUT_DIR = PROJECT_ROOT / "data" / "images"

ONEDRIVE_FOLDER = "/Desktop/Documents/PLANS"
GRAPH_BASE = "https://graph.microsoft.com/v1.0"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
GRAPH_EXPLORER_URL = "https://developer.microsoft.com/en-us/graph/graph-explorer"


def _load_env() -> None:
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


# ---------------------------------------------------------------------------
# Auth -- two modes:
#   1. Client-credentials (preferred): ACORN_TENANT_ID + ACORN_CLIENT_ID +
#      ACORN_CLIENT_SECRET in .env (OUTLOOK_* names also accepted as aliases).
#      App-only token, hits /users/{upn}/drive/ where upn defaults to
#      plans@acornasbestos.co.uk (override via ACORN_USER_PRINCIPAL_NAME).
#      Requires Files.Read.All *application* permission with admin consent
#      in Azure Portal.
#   2. Personal token (fallback): MSGRAPH_TOKEN pasted from Graph Explorer.
#      Hits /me/drive/. Token expires every ~hour.
# ---------------------------------------------------------------------------

def _save_token_to_env(token: str) -> None:
    existing = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
    out = [ln for ln in existing if not ln.startswith("MSGRAPH_TOKEN=")]
    out.append(f"MSGRAPH_TOKEN={token}")
    ENV_FILE.write_text("\n".join(out) + "\n", encoding="utf-8")


def _prompt_for_token() -> str:
    print()
    print("=" * 72)
    print("  Microsoft Graph access token required")
    print("=" * 72)
    print(f"  1. Open: {GRAPH_EXPLORER_URL}")
    print("  2. Sign in with the Acorn account that owns the PLANS folder.")
    print("  3. Modify permissions -> consent to 'Files.Read'.")
    print("  4. Run any sample query (e.g. GET /me) so a token is minted.")
    print("  5. Click your avatar -> 'Access token' -> copy.")
    print("=" * 72)
    if not sys.stdin.isatty():
        print("  [ERROR] No MSGRAPH_TOKEN in .env and stdin isn't interactive.")
        print("          Put MSGRAPH_TOKEN=<token> in .env and re-run.")
        sys.exit(1)
    token = input("Paste token here: ").strip()
    if not token:
        print("  [ERROR] Empty token.")
        sys.exit(1)
    _save_token_to_env(token)
    print("  Token saved to .env.")
    return token


def _acquire_app_token() -> Optional[str]:
    """Try client-credentials flow. Returns token or None on misconfig."""
    tenant = (os.environ.get("ACORN_TENANT_ID")
              or os.environ.get("OUTLOOK_TENANT_ID")
              or "").strip()
    client_id = (os.environ.get("ACORN_CLIENT_ID")
                 or os.environ.get("OUTLOOK_CLIENT_ID")
                 or "").strip()
    secret = (os.environ.get("ACORN_CLIENT_SECRET")
              or os.environ.get("OUTLOOK_CLIENT_SECRET")
              or "").strip()
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
    except requests.RequestException as e:
        print(f"  [AUTH] Token request network error: {e}")
        return None
    if r.status_code != 200:
        print(f"  [AUTH] Client-credentials token request failed: HTTP {r.status_code}")
        print(f"         Response: {r.text[:400]}")
        return None
    return r.json().get("access_token")


# Resolves to one of:
#   "app"      -> token, target_drive_prefix=/users/{upn}/drive
#   "user"     -> token, target_drive_prefix=/me/drive
_AUTH_MODE: Optional[str] = None
_DRIVE_PREFIX: Optional[str] = None


def _get_token() -> str:
    """Decide auth mode and return a working token."""
    global _AUTH_MODE, _DRIVE_PREFIX

    app_token = _acquire_app_token()
    if app_token:
        upn = (os.environ.get("ACORN_USER_PRINCIPAL_NAME")
               or os.environ.get("MS_USER_PRINCIPAL_NAME")
               or "plans@acornasbestos.co.uk").strip()
        _AUTH_MODE = "app"
        _DRIVE_PREFIX = f"/users/{upn}/drive"
        print(f"  [AUTH] Mode: client-credentials (app-only)")
        print(f"  [AUTH] Target drive: {_DRIVE_PREFIX}")
        return app_token

    # Fallback: pasted user token
    tok = os.environ.get("MSGRAPH_TOKEN", "").strip()
    if not tok:
        tok = _prompt_for_token()
    _AUTH_MODE = "user"
    _DRIVE_PREFIX = "/me/drive"
    print(f"  [AUTH] Mode: pasted Graph Explorer token (/me/drive)")
    return tok


def _graph_get(url: str, token: str) -> dict:
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=30)
    if resp.status_code == 401:
        if _AUTH_MODE == "app":
            # App tokens shouldn't 401 mid-run unless they expired (~1hr).
            # Re-acquire silently.
            print("  [AUTH] App token expired, re-acquiring...")
            new_token = _acquire_app_token()
            if not new_token:
                print("  [ERROR] Re-acquire failed.")
                sys.exit(1)
            os.environ["MSGRAPH_TOKEN"] = new_token  # cache for callers
            resp = requests.get(url, headers={"Authorization": f"Bearer {new_token}"}, timeout=30)
        else:
            print("  [AUTH] Token expired or invalid.")
            new_token = _prompt_for_token()
            resp = requests.get(url, headers={"Authorization": f"Bearer {new_token}"}, timeout=30)
            if resp.status_code == 401:
                print("  [ERROR] New token also rejected. Check Files.Read consent.")
                sys.exit(1)
            os.environ["MSGRAPH_TOKEN"] = new_token
    if resp.status_code == 403:
        print(f"  [403] {url}")
        print(f"        Response: {resp.text[:400]}")
        if _AUTH_MODE == "app":
            print()
            print("  [HINT] App-only auth needs APPLICATION permissions (not delegated).")
            print("         Azure Portal -> Your app -> API permissions ->")
            print("         Microsoft Graph -> Application permissions -> Files.Read.All")
            print("         then 'Grant admin consent'. Delegated permissions won't work")
            print("         for client-credentials flow.")
        sys.exit(1)
    resp.raise_for_status()
    return resp.json()


def _iter_children(folder_path: str, token: str) -> Iterator[dict]:
    """Yield every driveItem under folder_path, recursing into subfolders."""
    safe_path = folder_path.lstrip("/")
    url: str | None = f"{GRAPH_BASE}{_DRIVE_PREFIX}/root:/{safe_path}:/children?$top=200"
    while url:
        data = _graph_get(url, token)
        for item in data.get("value", []):
            if "folder" in item:
                sub = f"{folder_path}/{item['name']}"
                yield from _iter_children(sub, token)
            else:
                yield item
        url = data.get("@odata.nextLink")


def _is_image(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_EXTS


def _download_file(item: dict, dest: Path) -> int:
    """Stream-download a driveItem via @microsoft.graph.downloadUrl.

    The downloadUrl is a pre-signed short-lived URL; no auth header needed
    (and adding one actually trips a 401 in some tenants).
    """
    url = item.get("@microsoft.graph.downloadUrl")
    if not url:
        raise RuntimeError(f"no downloadUrl on item {item.get('name')}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.replace(dest)
    return dest.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after N new downloads (smoke test).")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would be downloaded, no writes.")
    args = parser.parse_args()

    _load_env()
    token = _get_token()
    os.environ["MSGRAPH_TOKEN"] = token

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Target folder (OneDrive): {ONEDRIVE_FOLDER}")
    print(f"  Output (local):           {OUTPUT_DIR}")
    print(f"  Extensions:               {sorted(IMAGE_EXTS)}")
    print()

    scanned = 0
    skipped_nonimage = 0
    skipped_existing = 0
    downloaded = 0
    errors = 0
    total_bytes = 0

    try:
        for item in _iter_children(ONEDRIVE_FOLDER, token):
            scanned += 1
            name = item.get("name", "")
            if not _is_image(name):
                skipped_nonimage += 1
                continue

            dest = OUTPUT_DIR / name
            # If name collides across subfolders, keep the parent in the filename.
            parent = (item.get("parentReference", {}) or {}).get("path", "")
            if dest.exists() and item.get("size") != dest.stat().st_size:
                # Different file with same name — disambiguate
                tag = parent.split("PLANS/")[-1].replace("/", "_") if "PLANS/" in parent else "alt"
                dest = OUTPUT_DIR / f"{Path(name).stem}__{tag}{Path(name).suffix}"

            if dest.exists():
                skipped_existing += 1
                continue

            size_kb = round((item.get("size") or 0) / 1024, 1)
            if args.dry_run:
                print(f"  [DRY] would download: {name} ({size_kb} KB)")
                downloaded += 1
            else:
                print(f"  [{downloaded + 1}] {name} ({size_kb} KB) ...", end=" ", flush=True)
                try:
                    n = _download_file(item, dest)
                    total_bytes += n
                    downloaded += 1
                    print("ok")
                except Exception as e:
                    errors += 1
                    print(f"FAILED ({type(e).__name__}: {e})")

            if args.limit and downloaded >= args.limit:
                print(f"\n  [LIMIT] Stopping after {args.limit} new files.")
                break
    except requests.HTTPError as e:
        print(f"\n  [ERROR] Graph API call failed: {e}")
        print(f"          Response: {e.response.text[:500] if e.response else 'no body'}")
        sys.exit(1)

    print()
    print("=" * 72)
    print("  DOWNLOAD COMPLETE")
    print("=" * 72)
    print(f"  Items scanned in OneDrive:   {scanned}")
    print(f"  Skipped (non-image):         {skipped_nonimage}")
    print(f"  Skipped (already local):     {skipped_existing}")
    print(f"  Downloaded:                  {downloaded}")
    print(f"  Errors:                      {errors}")
    print(f"  Total bytes:                 {total_bytes / 1024 / 1024:.1f} MB")
    print(f"  Output folder:               {OUTPUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
