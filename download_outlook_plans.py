"""Pull plan-image attachments straight from the plans@acornasbestos.co.uk
shared mailbox into data/images/.

Mirrors the filter used by the n8n "Email-attachments -> OneDrive" workflow:
    - size > 500 KB
    - name does NOT match /^image\\d+/i   (skips inline signature thumbnails)
    - extension in {.jpg, .jpeg, .png, .heic}

Auth: same ACORN_* client-credentials as download_plans.py. App needs
Mail.Read application permission in Azure AD (already present per
plans/outlook_to_tracker.py and a successful probe).

Resumable: progress is saved per-message to data/outlook_progress.json
so re-running picks up where it left off. Already-downloaded names are
skipped by file-existence check.

Usage:
    python download_outlook_plans.py                  # pull everything
    python download_outlook_plans.py --limit 100      # first 100 messages
    python download_outlook_plans.py --since 2025-01  # since YYYY-MM[-DD]
    python download_outlook_plans.py --dry-run        # list, no downloads
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import hashlib
from pathlib import Path
from typing import Iterator, Optional

import requests

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "images"
PROGRESS_FILE = PROJECT_ROOT / "data" / "outlook_progress.json"

MAILBOX = "plans@acornasbestos.co.uk"
GRAPH_BASE = "https://graph.microsoft.com/v1.0"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic"}
MIN_SIZE_BYTES = 500 * 1024
INLINE_SIG_RE = re.compile(r"^image\d+", re.IGNORECASE)

PAGE_SIZE = 100  # messages per page


# ---------------------------------------------------------------------------
# Reuse the auth helpers from download_plans.py (already loads .env, gets app
# token via client-credentials).
# ---------------------------------------------------------------------------

sys.path.insert(0, str(PROJECT_ROOT))
from download_plans import _load_env, _acquire_app_token  # noqa: E402


# Module-level token cache. App-only tokens live ~60 min; we refresh ~5 min
# before expiry, and also on any 401 (Graph occasionally rejects tokens
# slightly before their stated TTL).
_TOKEN: Optional[str] = None
_TOKEN_EXPIRES_AT: float = 0.0
_TOKEN_TTL_SAFETY_SEC = 300  # refresh 5 min before nominal expiry


def _get_token(force_refresh: bool = False) -> Optional[str]:
    global _TOKEN, _TOKEN_EXPIRES_AT
    now = time.time()
    if not force_refresh and _TOKEN and now < _TOKEN_EXPIRES_AT - _TOKEN_TTL_SAFETY_SEC:
        return _TOKEN
    tok = _acquire_app_token()
    if not tok:
        return None
    _TOKEN = tok
    # _acquire_app_token doesn't expose expires_in; assume the documented 3600s.
    _TOKEN_EXPIRES_AT = now + 3600
    if force_refresh:
        print("  [AUTH] Refreshed Graph access token.")
    return _TOKEN


def _graph_get(url: str, retries: int = 4) -> Optional[dict]:
    """GET with auto-refresh of expired tokens. 'eventual' consistency +
    $count=true is required for combining $filter on Outlook messages;
    otherwise Graph returns InefficientFilter 400."""
    refreshed_once = False
    for attempt in range(retries):
        tok = _get_token()
        if not tok:
            print("  [AUTH] Token acquisition failed.")
            return None
        headers = {
            "Authorization": f"Bearer {tok}",
            "ConsistencyLevel": "eventual",
        }
        try:
            r = requests.get(url, headers=headers, timeout=45)
        except requests.RequestException as e:
            print(f"  [NET] {e} (attempt {attempt + 1}/{retries})")
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 401 and not refreshed_once:
            print("  [AUTH] 401 - refreshing token and retrying.")
            _get_token(force_refresh=True)
            refreshed_once = True
            continue
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 10))
            print(f"  [429] backing off {wait}s")
            time.sleep(wait)
            continue
        if 400 <= r.status_code < 500:
            print(f"  [HTTP {r.status_code}] {url}\n    {r.text[:250]}")
            return None
        print(f"  [HTTP {r.status_code}] {url}\n    {r.text[:200]}")
        time.sleep(1 + attempt)
    return None


def _graph_get_binary(url: str, retries: int = 4) -> Optional[bytes]:
    refreshed_once = False
    for attempt in range(retries):
        tok = _get_token()
        if not tok:
            return None
        headers = {"Authorization": f"Bearer {tok}"}
        try:
            r = requests.get(url, headers=headers, timeout=120)
        except requests.RequestException as e:
            print(f"    [NET] {e} (attempt {attempt + 1}/{retries})")
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 200:
            return r.content
        if r.status_code == 401 and not refreshed_once:
            _get_token(force_refresh=True)
            refreshed_once = True
            continue
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 10))
            time.sleep(wait)
            continue
        if 400 <= r.status_code < 500:
            print(f"    [HTTP {r.status_code}] {r.text[:200]}")
            return None
        print(f"    [HTTP {r.status_code}] {r.text[:200]}")
        time.sleep(1 + attempt)
    return None


# ---------------------------------------------------------------------------
# Iterators
# ---------------------------------------------------------------------------

def _iter_messages(since: Optional[str] = None) -> Iterator[dict]:
    """Yield messages with attachments inlined via $expand=attachments.

    This is the perf hot path: with $expand we get up to PAGE_SIZE messages
    AND their attachment metadata in a single call, instead of 1+N calls per
    page. We don't ask for contentBytes (would inflate response + trip a
    size limit when any attachment exceeds 3 MB), so we still need a
    separate per-attachment download call later -- but only for matches.
    """
    filt = "hasAttachments eq true"
    if since:
        filt += f" and receivedDateTime ge {since}T00:00:00Z"
    select = "id,subject,receivedDateTime,from"
    expand = "attachments($select=id,name,size,contentType)"
    url: Optional[str] = (
        f"{GRAPH_BASE}/users/{MAILBOX}/messages"
        f"?$filter={filt}"
        f"&$select={select}"
        f"&$expand={expand}"
        f"&$count=true"
        f"&$top={PAGE_SIZE}"
    )
    while url:
        data = _graph_get(url)
        if not data:
            return
        for msg in data.get("value", []):
            yield msg
        url = data.get("@odata.nextLink")


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _is_plan_attachment(att: dict) -> bool:
    if att.get("@odata.type") != "#microsoft.graph.fileAttachment":
        return False
    name = (att.get("name") or "").strip()
    if not name:
        return False
    if INLINE_SIG_RE.match(name):
        return False
    if Path(name).suffix.lower() not in IMAGE_EXTS:
        return False
    if (att.get("size") or 0) < MIN_SIZE_BYTES:
        return False
    return True


def _safe_filename(name: str) -> str:
    """Replace characters Windows / common filesystems don't accept."""
    bad = '<>:"/\\|?*'
    out = "".join("_" if c in bad else c for c in name)
    return out.strip().rstrip(".")


def _attachment_filename(att: dict, msg: dict) -> str:
    """Return a stable filename that avoids cross-message collisions."""
    original = _safe_filename(att.get("name") or "attachment")
    digest = hashlib.sha1(
        f"{msg.get('id', '')}:{att.get('id', '')}".encode("utf-8")
    ).hexdigest()[:10]
    stem = Path(original).stem
    suffix = Path(original).suffix
    return f"{stem}__{digest}{suffix}"


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

def _load_progress() -> dict:
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_progress(p: dict) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = PROGRESS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(p, indent=2), encoding="utf-8")
    tmp.replace(PROGRESS_FILE)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after scanning N messages (smoke test).")
    ap.add_argument("--max-downloads", type=int, default=10000,
                    help="Stop after N successful new downloads. "
                         "Counts only files actually written this run "
                         "(skips and dry-runs don't count). Default 10000.")
    ap.add_argument("--since", default=None,
                    help="Only messages received on/after YYYY-MM-DD.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List matching attachments; don't download.")
    args = ap.parse_args()

    _load_env()
    if not _get_token():
        print("[AUTH] Could not acquire app token. Check ACORN_* in .env.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    progress = _load_progress()

    print(f"  Mailbox:       {MAILBOX}")
    print(f"  Output:        {OUTPUT_DIR}")
    print(f"  Filter:        ext in {sorted(IMAGE_EXTS)}, size > {MIN_SIZE_BYTES // 1024} KB,")
    print(f"                 name not matching /^image\\d+/i (inline sigs)")
    print(f"  Resume state:  {len(progress)} messages already processed")
    print(f"  Max downloads: {args.max_downloads}")
    if args.since:  print(f"  Since:         {args.since}")
    if args.limit:  print(f"  Limit:         {args.limit} messages")
    if args.dry_run:print(f"  Dry run:       yes")
    print()

    msgs_seen = 0
    msgs_with_match = 0
    matched_atts = 0
    downloaded = 0
    skipped_existing = 0
    errors = 0
    total_bytes = 0

    stopped_early = False
    try:
        for msg in _iter_messages(since=args.since):
            msgs_seen += 1
            msg_id = msg["id"]

            if msg_id in progress and progress[msg_id].get("status") != "dry_run":
                # Already scanned this message in a previous run.
                continue

            # Attachments are inlined via $expand in _iter_messages.
            attachments = msg.get("attachments") or []
            plan_atts = [a for a in attachments if _is_plan_attachment(a)]

            if not plan_atts:
                # Note the message as processed so we don't re-fetch attachments.
                if not args.dry_run:
                    progress[msg_id] = {"status": "no_plans", "attachments": 0}
            else:
                msgs_with_match += 1
                row_entries = []
                for att in plan_atts:
                    matched_atts += 1
                    name = _attachment_filename(att, msg)
                    dest = OUTPUT_DIR / name
                    size_kb = round(att["size"] / 1024, 1)

                    if dest.exists():
                        skipped_existing += 1
                        row_entries.append({"name": name, "status": "already_local"})
                        continue

                    if args.dry_run:
                        print(f"  [DRY] {name} ({size_kb} KB) - msg {msg.get('subject','')[:40]!r}")
                        row_entries.append({"name": name, "status": "dry_run"})
                        continue

                    url = (f"{GRAPH_BASE}/users/{MAILBOX}/messages/{msg_id}"
                           f"/attachments/{att['id']}/$value")
                    print(f"  [{downloaded + 1}] {name} ({size_kb} KB) ...", end=" ", flush=True)
                    payload = _graph_get_binary(url)
                    if not payload:
                        errors += 1
                        print("FAILED")
                        row_entries.append({"name": name, "status": "download_failed"})
                        continue

                    tmp = dest.with_suffix(dest.suffix + ".part")
                    tmp.write_bytes(payload)
                    tmp.replace(dest)
                    downloaded += 1
                    total_bytes += len(payload)
                    print(f"ok ({round(len(payload)/1024,1)} KB)")
                    row_entries.append({"name": name, "status": "downloaded",
                                        "size": len(payload)})

                progress[msg_id] = {
                    "status": "dry_run" if args.dry_run else "processed",
                    "subject": (msg.get("subject") or "")[:120],
                    "received": msg.get("receivedDateTime"),
                    "attachments": len(plan_atts),
                    "items": row_entries,
                }

            # Save progress every 25 messages.
            if msgs_seen % 25 == 0 and not args.dry_run:
                _save_progress(progress)
                print(f"  ... scanned {msgs_seen} msgs, matched {msgs_with_match}, "
                      f"downloaded {downloaded}, errors {errors}")

            if args.limit and msgs_seen >= args.limit:
                print(f"\n  [LIMIT] Reached --limit {args.limit}, stopping.")
                stopped_early = True
                break
            if downloaded >= args.max_downloads:
                print(f"\n  [CAP] Reached --max-downloads {args.max_downloads}, stopping.")
                stopped_early = True
                break
    except KeyboardInterrupt:
        print("\n  [INTERRUPTED] Saving progress before exit.")
    finally:
        if not args.dry_run:
            _save_progress(progress)

    print()
    print("=" * 72)
    print("  OUTLOOK DOWNLOAD COMPLETE")
    print("=" * 72)
    print(f"  Messages scanned:        {msgs_seen}")
    print(f"  Messages w/ plan attach: {msgs_with_match}")
    print(f"  Plan attachments seen:   {matched_atts}")
    print(f"  Downloaded this run:     {downloaded}")
    print(f"  Already on disk:         {skipped_existing}")
    print(f"  Errors:                  {errors}")
    print(f"  Total bytes:             {total_bytes / 1024 / 1024:.1f} MB")
    print(f"  Output folder:           {OUTPUT_DIR}")
    print(f"  Progress:                {PROGRESS_FILE}")
    print("=" * 72)


if __name__ == "__main__":
    main()
