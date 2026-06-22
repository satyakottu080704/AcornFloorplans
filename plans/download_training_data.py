#!/usr/bin/env python3
"""
Alpha Tracker - VSDX Plan Downloader
=====================================
Downloads all .vsdx floor plan files from Alpha Tracker projects.

Usage:
    # Test with 10 projects first
    python download_plans.py --limit 10

    # Download all completed projects
    python download_plans.py --status Complete

    # Download everything, custom output folder
    python download_plans.py --output C:\\training_data\\vsdx

    # Dry run - see what's available without downloading
    python download_plans.py --dry-run --limit 50

    # Download specific project
    python download_plans.py --project N-60269

Requirements:
    pip install requests
"""

import os
import csv
import sys
import time
import argparse
import requests
from pathlib import Path
from datetime import datetime

# ─── CONFIG ──────────────────────────────────────────────────────────────────
BASE_URL   = "https://manager.alphatracker.co.uk/api"
API_KEY    = os.getenv("ALPHA_TRACKER_API_KEY", "D5921962-7E4F-4C1F-8A8E-A62F8D21BF0B")
CLIENT_ID  = os.getenv("ALPHA_TRACKER_CLIENT_ID", "6361A8E1-F174-40D4-AD28-F3EF69A6EE2C")

HEADERS = {
    "x-api-key":  API_KEY,
    "client-id":  CLIENT_ID,
    "accept":     "application/json",
}

# File extensions to download (add others if needed)
TARGET_EXTENSIONS = {".vsdx", ".vsd"}

# Delay between API calls to avoid rate limiting (seconds)
API_DELAY = 0.3


# ─── API HELPERS ─────────────────────────────────────────────────────────────

def api_get(path: str, params: dict = None, retries: int = 3) -> dict | list | None:
    """Make a GET request to Alpha Tracker API with retry logic."""
    url = f"{BASE_URL}/{path.lstrip('/')}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"  [RATE LIMIT] Waiting {wait}s...")
                time.sleep(wait)
            elif resp.status_code == 404:
                return None  # Not found - don't retry
            else:
                print(f"  [HTTP {resp.status_code}] {url}")
                if attempt < retries - 1:
                    time.sleep(2)
        except requests.exceptions.Timeout:
            print(f"  [TIMEOUT] attempt {attempt+1}/{retries}: {url}")
            time.sleep(3)
        except requests.exceptions.ConnectionError as e:
            print(f"  [CONNECTION ERROR] {e}")
            time.sleep(5)
    return None


def api_download(path: str, params: dict, dest_path: Path) -> bool:
    """Download a file from Alpha Tracker API."""
    url = f"{BASE_URL}/{path.lstrip('/')}"
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=60, stream=True)
        if resp.status_code == 200:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        else:
            print(f"  [HTTP {resp.status_code}] Download failed: {url}")
            return False
    except Exception as e:
        print(f"  [ERROR] Download error: {e}")
        return False


# ─── CORE FUNCTIONS ──────────────────────────────────────────────────────────

def get_all_projects(status_filter: str = None, limit: int = None) -> list[dict]:
    """
    Fetch all projects from Alpha Tracker with optional status filter.
    Handles pagination automatically.
    """
    all_projects = []
    page = 1
    page_size = 100

    print(f"[PROJECTS] Fetching projects" + 
          (f" (status={status_filter})" if status_filter else "") + "...")

    while True:
        params = {
            "paginate":   "true",
            "pageNumber": page,
            "limit":      page_size,
        }
        if status_filter:
            params["filter"] = f"status='{status_filter}'"

        data = api_get("projects/search", params=params)
        time.sleep(API_DELAY)

        if not data:
            break

        # Handle both list response and paginated object response
        if isinstance(data, list):
            projects = data
            has_more = False
        elif isinstance(data, dict):
            projects = data.get("data", data.get("projects", data.get("items", [])))
            total    = data.get("total", data.get("totalCount", 0))
            has_more = (page * page_size) < total
        else:
            break

        if not projects:
            break

        all_projects.extend(projects)
        print(f"  Page {page}: got {len(projects)} projects (total so far: {len(all_projects)})")

        # Apply limit
        if limit and len(all_projects) >= limit:
            all_projects = all_projects[:limit]
            break

        if not has_more or len(projects) < page_size:
            break

        page += 1

    print(f"[PROJECTS] Found {len(all_projects)} projects total")
    return all_projects


def get_project_files(project_number: str) -> list[dict]:
    """
    Get all files attached to a project.
    Returns list of {filename, size, ...}
    """
    data = api_get(f"files/projects/{project_number}")
    time.sleep(API_DELAY)

    if not data:
        return []

    # Handle both list and dict responses
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return data.get("files", data.get("data", data.get("items", [])))

    return []


def extract_project_number(project: dict) -> str | None:
    """Extract project number from project dict."""
    for key in ["projectNumber", "project_number", "number", "id", "projectNo"]:
        val = project.get(key)
        if val:
            return str(val).strip()
    return None


def extract_filename(file_record: dict) -> str | None:
    """Extract filename from file record."""
    for key in ["fileName", "filename", "name", "file_name", "originalName"]:
        val = file_record.get(key)
        if val:
            return str(val).strip()
    return None


def is_target_file(filename: str) -> bool:
    """Check if filename is a target file type we want to download."""
    if not filename:
        return False
    ext = Path(filename).suffix.lower()
    return ext in TARGET_EXTENSIONS


# ─── MAIN DOWNLOAD LOGIC ─────────────────────────────────────────────────────

def download_vsdx_files(
    output_dir:    Path,
    status_filter: str  = None,
    limit:         int  = None,
    dry_run:       bool = False,
    project_filter: str = None,
) -> Path:
    """
    Main function: scan projects, find .vsdx files, download them.
    Returns path to the CSV report.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"download_report_{timestamp}.csv"

    rows = []
    downloaded = 0
    skipped    = 0
    not_found  = 0
    errors     = 0

    # ── Get projects ──
    if project_filter:
        # Single project mode
        projects = [{"projectNumber": project_filter}]
        print(f"[MODE] Single project: {project_filter}")
    else:
        projects = get_all_projects(status_filter=status_filter, limit=limit)

    if not projects:
        print("[ERROR] No projects found. Check API credentials.")
        return report_path

    print(f"\n[SCAN] Scanning {len(projects)} projects for .vsdx files...\n")

    for i, project in enumerate(projects, 1):
        project_number = extract_project_number(project)
        if not project_number:
            continue

        # Progress indicator
        print(f"[{i:4d}/{len(projects)}] {project_number}", end=" ")

        # Get files for this project
        files = get_project_files(project_number)

        if files is None:
            print(f"→ API error")
            rows.append({
                "project_number": project_number,
                "filename":       "",
                "size_kb":        "",
                "status":         "api_error",
                "output_path":    "",
                "error":          "Failed to get file list",
            })
            errors += 1
            continue

        # Find target files
        target_files = [f for f in files if is_target_file(extract_filename(f))]

        if not target_files:
            print(f"→ no .vsdx found ({len(files)} files total)")
            rows.append({
                "project_number": project_number,
                "filename":       "",
                "size_kb":        "",
                "status":         "no_vsdx",
                "output_path":    "",
                "error":          "",
            })
            not_found += 1
            continue

        # Download each target file
        for file_record in target_files:
            filename = extract_filename(file_record)
            size_bytes = file_record.get("fileSize", file_record.get("size", 0))
            size_kb = round(int(size_bytes or 0) / 1024, 1)

            # Output path: output_dir/project_number/filename.vsdx
            dest_path = output_dir / project_number / filename

            # Skip if already downloaded
            if dest_path.exists():
                print(f"→ already exists: {filename} ({size_kb}KB)")
                rows.append({
                    "project_number": project_number,
                    "filename":       filename,
                    "size_kb":        size_kb,
                    "status":         "already_exists",
                    "output_path":    str(dest_path),
                    "error":          "",
                })
                skipped += 1
                continue

            if dry_run:
                print(f"→ [DRY RUN] would download: {filename} ({size_kb}KB)")
                rows.append({
                    "project_number": project_number,
                    "filename":       filename,
                    "size_kb":        size_kb,
                    "status":         "dry_run",
                    "output_path":    str(dest_path),
                    "error":          "",
                })
                continue

            # Download the file
            print(f"→ downloading: {filename} ({size_kb}KB)...", end=" ")
            success = api_download(
                f"files/projects/{project_number}/download",
                params={"filename": filename},
                dest_path=dest_path,
            )

            if success:
                actual_size_kb = round(dest_path.stat().st_size / 1024, 1)
                print(f"✓ saved ({actual_size_kb}KB)")
                rows.append({
                    "project_number": project_number,
                    "filename":       filename,
                    "size_kb":        actual_size_kb,
                    "status":         "downloaded",
                    "output_path":    str(dest_path),
                    "error":          "",
                })
                downloaded += 1
            else:
                print(f"✗ failed")
                rows.append({
                    "project_number": project_number,
                    "filename":       filename,
                    "size_kb":        size_kb,
                    "status":         "download_failed",
                    "output_path":    "",
                    "error":          "Download request failed",
                })
                errors += 1

    # ── Write CSV report ──
    fieldnames = ["project_number", "filename", "size_kb", "status", "output_path", "error"]
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ── Summary ──
    print(f"""
{'='*60}
DOWNLOAD COMPLETE
{'='*60}
  Projects scanned:    {len(projects)}
  .vsdx files found:   {downloaded + skipped + errors}
  Downloaded:          {downloaded}
  Already existed:     {skipped}
  Not found:           {not_found}
  Errors:              {errors}
  Output folder:       {output_dir}
  Report:              {report_path}
{'='*60}
""")

    return report_path


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download .vsdx plan files from Alpha Tracker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with 10 projects first (always do this first!)
  python download_plans.py --limit 10

  # Download all completed projects
  python download_plans.py --status Complete

  # Dry run - see what's available without downloading
  python download_plans.py --dry-run --limit 100

  # Download a specific project
  python download_plans.py --project N-60269

  # Custom output folder
  python download_plans.py --output C:\\training_data\\vsdx --limit 50
        """
    )
    parser.add_argument(
        "--output", default="output/downloaded_plans",
        help="Output folder (default: output/downloaded_plans)"
    )
    parser.add_argument(
        "--status", default=None,
        help="Filter by project status e.g. Complete, Active, Invoiced"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max number of projects to scan"
    )
    parser.add_argument(
        "--project", default=None,
        help="Download a single specific project e.g. N-60269"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Scan only, show what would be downloaded, don't actually download"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)

    print(f"""
Alpha Tracker VSDX Downloader
==============================
API:     {BASE_URL}
Output:  {output_dir}
Status:  {args.status or 'all'}
Limit:   {args.limit or 'none'}
Project: {args.project or 'all'}
Dry run: {args.dry_run}
==============================
""")

    report = download_vsdx_files(
        output_dir=output_dir,
        status_filter=args.status,
        limit=args.limit,
        dry_run=args.dry_run,
        project_filter=args.project,
    )

    print(f"Report saved: {report}")


if __name__ == "__main__":
    main()
