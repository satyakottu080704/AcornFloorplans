#!/usr/bin/env python3
"""
Outlook to Visio to Tracker Pipeline (Microsoft Graph API)
===========================================================
Complete workflow:
1. Extract plan images from Outlook emails via Microsoft Graph API
2. Convert images to Visio plans
3. Upload both image and Visio to Alpha Tracker

Features:
- Smart image selection: prefers 'processed' filenames and largest resolution
- AI-powered Visio generation from images (Ollama LLaVA)
- Headless browser support for automated uploads to Alpha Tracker

Requirements:
- requests (HTTP client)
- Ollama with llava model (optional, for Visio generation)
- Microsoft Graph API credentials in .env
- Alpha Tracker API credentials in .env
- Pillow (optional, for image resolution checking)

Usage:
    # From Outlook email
    python outlook_to_tracker.py --project N-99752
    
    # From local image file
    python outlook_to_tracker.py --project N-99752 --image /path/to/plan.png
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import tempfile
import base64
import json
import time
from datetime import datetime, timedelta

# Handle imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from api import get_project, update_project, upload_file
except (ImportError, ValueError):
    try:
        from api import get_project, update_project, upload_file
    except ImportError:
        def upload_file(*args, **kwargs):
            return {"success": False, "error": "upload_file not available", "data": None}
        def get_project(*args, **kwargs):
            return {}
        def update_project(*args, **kwargs):
            return {"success": False, "error": "update_project not available"}

try:
    from utils.helpers import Colors
    from utils.config import get_config
except (ImportError, ValueError):
    from utils.helpers import Colors
    from utils.config import get_config

# Optional converter module (can be absent)
ImageToVizioPlanConverter = None
try:
    from .image_to_visio import ImageToVizioPlanConverter
except (ImportError, ValueError):
    try:
        from plans.image_to_visio import ImageToVizioPlanConverter
    except ImportError:
        ImageToVizioPlanConverter = None

# Import requests for Graph API
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Optional image library for resolution checks
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


# =============================================================================
# OUTLOOK INTEGRATION (Microsoft Graph API)
# =============================================================================

class OutlookPlansExtractor:
    """Extract plan images from Outlook emails via Microsoft Graph API."""
    
    GRAPH_API_BASE = "https://graph.microsoft.com/v1.0"
    OAUTH_TOKEN_URL = "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    
    def __init__(self, verbose: bool = True):
        """Initialize Outlook Graph API extractor."""
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests not available: pip install requests")
        
        self.verbose = verbose
        self.tenant_id = os.getenv("OUTLOOK_TENANT_ID")
        self.client_id = os.getenv("OUTLOOK_CLIENT_ID")
        self.client_secret = os.getenv("OUTLOOK_CLIENT_SECRET")
        self.outlook_email = os.getenv("OUTLOOK_EMAIL", "")
        
        if not all([self.tenant_id, self.client_id, self.client_secret]):
            raise ValueError(
                "Missing Outlook credentials in .env:\n"
                "  OUTLOOK_TENANT_ID\n"
                "  OUTLOOK_CLIENT_ID\n"
                "  OUTLOOK_CLIENT_SECRET"
            )
        
        self.access_token = None
        self.token_expires_at = None
        self._get_access_token()
        self.log(f"Outlook Graph API initialized for {self.outlook_email}", "SUCCESS")
    
    def _get_access_token(self):
        """Get fresh access token from Azure AD."""
        token_url = self.OAUTH_TOKEN_URL.format(tenant_id=self.tenant_id)
        
        payload = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "client_credentials",
            "scope": "https://graph.microsoft.com/.default"
        }
        
        try:
            response = requests.post(token_url, data=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            self.access_token = data["access_token"]
            expires_in = data.get("expires_in", 3600)
            self.token_expires_at = datetime.now() + timedelta(seconds=expires_in - 60)
            self.log(f"Access token acquired (expires at {self.token_expires_at})", "DEBUG")
        except Exception as e:
            self.log(f"Failed to get access token: {str(e)}", "ERROR")
            raise
    
    def _ensure_token_valid(self):
        """Refresh token if expired."""
        if not self.token_expires_at or datetime.now() >= self.token_expires_at:
            self.log("Token expired, refreshing...", "DEBUG")
            self._get_access_token()
    
    def _graph_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make authenticated request to Graph API with retries."""
        self._ensure_token_valid()

        url = f"{self.GRAPH_API_BASE}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }

        last_error = None
        for attempt in range(1, 4):
            try:
                response = requests.request(method, url, headers=headers, timeout=45, **kwargs)
                response.raise_for_status()
                return response.json() if response.content else {}
            except Exception as e:
                last_error = e
                self.log(f"Graph API error (attempt {attempt}/3): {str(e)}", "ERROR")
                time.sleep(2 * attempt)

        raise last_error
    
    def log(self, message: str, level: str = "INFO"):
        """Print log message"""
        if not self.verbose:
            return
        
        color = {
            "INFO": Colors.CYAN,
            "SUCCESS": Colors.GREEN,
            "WARNING": Colors.YELLOW,
            "ERROR": Colors.RED,
            "DEBUG": Colors.BLUE
        }.get(level, Colors.CYAN)
        
        print(f"{color}[{level}]{Colors.RESET} {message}")
    
    def find_emails_with_project(
        self,
        project_number: str,
        folder_name: str = "Inbox",
        subject_hint: Optional[str] = None
    ) -> List[Dict]:
        """Find emails containing project number via Graph API."""
        self.log(f"Searching for emails about {project_number}")
        
        try:
            # Get messages from the folder (fetch recent and filter locally)
            folder = folder_name or "inbox"
            endpoint = f"/users/{self.outlook_email}/mailFolders/{folder}/messages"
            params = {"$top": 200, "$select": "subject,bodyPreview,from,receivedDateTime,id"}

            messages = self._graph_request("GET", endpoint, params=params)

            emails = []
            for msg in messages.get("value", []):
                email_id = msg["id"]
                subject = msg.get("subject", "")
                body_preview = (msg.get("bodyPreview") or "")

                # Local filter: check subject or body preview for project number
                if project_number.lower() not in subject.lower() and project_number.lower() not in body_preview.lower():
                    continue

                # Optional stricter subject filter
                if subject_hint and subject_hint.lower() not in subject.lower():
                    continue

                # Get attachments for this message
                attachments_response = self._graph_request(
                    "GET",
                    f"/users/{self.outlook_email}/messages/{email_id}/attachments"
                )
                
                plan_attachments = self._filter_plan_attachments(
                    attachments_response.get("value", [])
                )
                
                if plan_attachments:
                    emails.append({
                        "subject": subject,
                        "sender": msg.get("from", {}).get("emailAddress", {}).get("name", "Unknown"),
                        "received": msg.get("receivedDateTime", ""),
                        "attachments": plan_attachments,
                        "message_id": email_id
                    })
                    self.log(f"Found email: {subject[:50]}")
            
            self.log(f"Found {len(emails)} emails with plan attachments in {folder}", "SUCCESS")
            return emails
            
        except Exception as e:
            self.log(f"Error searching emails: {str(e)}", "ERROR")
            raise
    
    def _filter_plan_attachments(self, attachments: List[Dict]) -> List[Dict]:
        """Filter plan image attachments."""
        plan_extensions = ['.png', '.jpg', '.jpeg', '.pdf', '.tif', '.tiff']
        plan_keywords = ['plan', 'survey', 'floor', 'layout', 'map', 'asbestos']
        
        filtered = []
        for attachment in attachments:
            filename = attachment.get("name", "").lower()
            
            if attachment.get("@odata.type") != "#microsoft.graph.fileAttachment":
                continue
            
            has_plan_ext = any(filename.endswith(ext) for ext in plan_extensions)
            has_plan_keyword = any(kw in filename for kw in plan_keywords)
            
            if has_plan_ext and (has_plan_keyword or has_plan_ext):
                filtered.append({
                    "filename": attachment.get("name", ""),
                    "size": attachment.get("size", 0),
                    "attachment_id": attachment.get("id", "")
                })
                self.log(f"Plan attachment found: {attachment.get('name', '')}")
        
        return filtered
    
    def extract_attachment(self, email_item: Dict, 
                          output_folder: Optional[str] = None) -> List[str]:
        """Extract plan attachments from email via Graph API."""
        if not output_folder:
            output_folder = tempfile.gettempdir()
        
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        extracted_files = []
        message_id = email_item.get("message_id")
        
        for attachment in email_item.get("attachments", []):
            try:
                attachment_id = attachment.get("attachment_id")
                filename = attachment.get("filename")
                
                self.log(f"Downloading attachment: {filename}")
                
                # Get attachment content
                endpoint = f"/users/{self.outlook_email}/messages/{message_id}/attachments/{attachment_id}/$value"
                
                # Make raw request for binary content
                self._ensure_token_valid()
                url = f"{self.GRAPH_API_BASE}{endpoint}"
                headers = {"Authorization": f"Bearer {self.access_token}"}
                
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                
                # Save to file
                output_path = os.path.join(output_folder, filename)
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                extracted_files.append(output_path)
                self.log(f"Saved: {output_path}", "SUCCESS")
                
            except Exception as e:
                self.log(f"Failed to extract {attachment.get('filename')}: {str(e)}", "ERROR")
        
        return extracted_files


# =============================================================================
# ALPHA TRACKER FILE UPLOAD
# =============================================================================

class AlphaTrackerFileUploader:
    """Upload files to Alpha Tracker."""
    
    def __init__(self, verbose: bool = True):
        """Initialize uploader."""
        self.verbose = verbose
    
    def log(self, message: str, level: str = "INFO"):
        """Print log message"""
        if not self.verbose:
            return
        
        color = {
            "INFO": Colors.CYAN,
            "SUCCESS": Colors.GREEN,
            "WARNING": Colors.YELLOW,
            "ERROR": Colors.RED
        }.get(level, Colors.CYAN)
        
        print(f"{color}[{level}]{Colors.RESET} {message}")
    
    def upload_file_to_project(self, project_number: str, file_path: str,
                               file_type: str = "plan_image") -> Dict[str, Any]:
        """
        Upload file to Alpha Tracker project.
        
        Args:
            project_number: Project number
            file_path: Path to file to upload
            file_type: Type of file (plan_image_source, survey_plan_visio, etc.)
            
        Returns:
            Upload result
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            self.log(f"Uploading {file_type} to {project_number}: {file_path.name}")

            api_result = upload_file(project_number, str(file_path), file_type=file_type)
            if api_result.get("success"):
                result = {
                    "status": "success",
                    "project_number": project_number,
                    "filename": file_path.name,
                    "file_type": file_type,
                    "file_size": len(file_path.read_bytes()),
                    "uploaded_at": str(Path(file_path).stat().st_mtime),
                    "response": api_result.get("data"),
                }
                self.log(f"Uploaded {file_path.name}", "SUCCESS")
                return result

            error_msg = api_result.get("error") or "Unknown upload error"
            self.log(f"Upload failed: {error_msg}", "ERROR")
            return {
                "status": "error",
                "project_number": project_number,
                "filename": file_path.name,
                "file_type": file_type,
                "error": error_msg,
            }
            
        except Exception as e:
            self.log(f"Upload failed: {str(e)}", "ERROR")
            return {
                "status": "error",
                "project_number": project_number,
                "error": str(e)
            }


# =============================================================================
# COMPLETE PIPELINE
# =============================================================================

class OutlookToTrackerPipeline:
    """Complete workflow: Outlook -> Image -> Visio -> Tracker"""
    
    def __init__(self, client_type: str = "cardtronics", verbose: bool = True):
        """Initialize pipeline."""
        self.client_type = client_type.lower()
        self.verbose = verbose
        
        try:
            self.outlook_extractor = OutlookPlansExtractor(verbose=verbose)
        except ImportError:
            self.outlook_extractor = None
            self.log("Outlook extractor not available", "WARNING")
        
        self.file_uploader = AlphaTrackerFileUploader(verbose=verbose)
    
    def log(self, message: str, level: str = "INFO"):
        """Print log message"""
        if not self.verbose:
            return
        
        color = {
            "INFO": Colors.CYAN,
            "SUCCESS": Colors.GREEN,
            "WARNING": Colors.YELLOW,
            "ERROR": Colors.RED
        }.get(level, Colors.CYAN)
        
        print(f"{color}[{level}]{Colors.RESET} {message}")
    
    def _select_best_image(self, files: List[str], project_number: str) -> Optional[str]:
        """
        Choose the most suitable image from a list.
        
        Preference order:
        1. Filename contains project number
        2. Filename contains 'processed'
        3. Largest image by pixel area (requires Pillow) or file size
        """
        if not files:
            return None

        # 1) prefer filename containing project number
        for f in files:
            try:
                if project_number.lower() in Path(f).name.lower():
                    self.log(f"Selecting image with project number in name: {Path(f).name}")
                    return f
            except Exception:
                continue

        # 2) prefer processed variants
        for f in files:
            if 'processed' in Path(f).name.lower():
                self.log(f"Selecting processed image: {Path(f).name}")
                return f

        # 3) choose largest by resolution if available, else file size
        best = None
        best_metric = 0
        best_name = None
        
        for f in files:
            try:
                if PIL_AVAILABLE:
                    with Image.open(f) as im:
                        metric = im.width * im.height
                else:
                    metric = os.path.getsize(f)
            except Exception:
                try:
                    metric = os.path.getsize(f)
                except Exception:
                    metric = 0

            if metric > best_metric:
                best_metric = metric
                best = f
                best_name = Path(f).name

        if best:
            self.log(f"Selecting largest image: {best_name} ({best_metric} {'px' if PIL_AVAILABLE else 'bytes'})")
        
        return best
    
    def process_outlook_email(self, project_number: str, 
                             email_subject_hint: Optional[str] = None,
                             output_folder: Optional[str] = None) -> Dict[str, Any]:
        """Process email from Outlook: extract image, convert to Visio, upload to Tracker."""
        self.log(f"Starting Outlook to Tracker pipeline for {project_number}")
        
        if not self.outlook_extractor:
            raise RuntimeError("Outlook not available")
        
        if not output_folder:
            output_folder = Path(__file__).parent.parent / "output" / "plans"
        
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Find and extract from Outlook
        self.log("Step 1: Extracting images from Outlook email...")
        
        emails = []
        folders_to_search = ["Inbox", "Archive", "SentItems"]
        for folder_name in folders_to_search:
            emails = self.outlook_extractor.find_emails_with_project(
                project_number,
                folder_name=folder_name,
                subject_hint=email_subject_hint
            )
            if emails:
                break
        
        if not emails:
            return {
                "status": "error",
                "error": f"No emails found for {project_number}"
            }
        
        # Extract from first matching email
        email = emails[0]
        extracted_files = self.outlook_extractor.extract_attachment(
            email, 
            output_folder / "extracted"
        )
        
        if not extracted_files:
            return {
                "status": "error",
                "error": "No plan images extracted from email"
            }
        
        # Select best image among extracted files
        image_path = self._select_best_image(extracted_files, project_number)
        if not image_path:
            image_path = extracted_files[0]
        
        self.log(f"Selected image for conversion: {image_path}", "SUCCESS")
        
        # Step 2: Convert image to Visio
        self.log("Step 2: Converting image to Visio plan...")
        
        try:
            if ImageToVizioPlanConverter is None:
                raise ImportError("ImageToVizioPlanConverter module not available")
            converter = ImageToVizioPlanConverter(
                project_number=project_number,
                client_type=self.client_type,
                verbose=self.verbose
            )
            
            visio_path = output_folder / f"{project_number}.vsdx"
            conversion_result = converter.convert_image_to_visio(
                image_path=image_path,
                output_path=str(visio_path)
            )
            
            if conversion_result["status"] != "success":
                return {
                    "status": "error",
                    "error": "Image to Visio conversion failed"
                }
            
            self.log(f"Generated Visio: {visio_path}", "SUCCESS")
            
        except ImportError:
            self.log("Visio conversion skipped (Visio not available)", "WARNING")
            visio_path = None
        
        # Step 3: Upload to Alpha Tracker
        self.log("Step 3: Uploading files to Alpha Tracker...")
        
        upload_results = []
        
        # Upload original image
        image_upload = self.file_uploader.upload_file_to_project(
            project_number=project_number,
            file_path=image_path,
            file_type="plan_image_source"
        )
        upload_results.append(image_upload)
        
        # Upload Visio plan if generated
        if visio_path:
            visio_upload = self.file_uploader.upload_file_to_project(
                project_number=project_number,
                file_path=str(visio_path),
                file_type="survey_plan_visio"
            )
            upload_results.append(visio_upload)
        
        return {
            "status": "success",
            "project_number": project_number,
            "image_path": image_path,
            "visio_path": str(visio_path) if visio_path else None,
            "email": {
                "subject": email["subject"],
                "sender": email["sender"],
                "received": str(email["received"])
            },
            "conversion": conversion_result if visio_path else None,
            "uploads": upload_results
        }
    
    def process_plan_image(self, project_number: str, image_path: str,
                          output_folder: Optional[str] = None) -> Dict[str, Any]:
        """Process plan image file: convert to Visio, upload to Tracker."""
        self.log(f"Processing plan image for {project_number}")
        
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        if not output_folder:
            output_folder = Path(__file__).parent.parent / "output" / "plans"
        
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Convert to Visio
        self.log("Step 1: Converting image to Visio...")
        
        try:
            if ImageToVizioPlanConverter is None:
                raise ImportError("ImageToVizioPlanConverter module not available")
            converter = ImageToVizioPlanConverter(
                project_number=project_number,
                client_type=self.client_type,
                verbose=self.verbose
            )
            
            visio_path = output_folder / f"{project_number}.vsdx"
            conversion_result = converter.convert_image_to_visio(
                image_path=str(image_path),
                output_path=str(visio_path)
            )
            
            if conversion_result["status"] != "success":
                return {
                    "status": "error",
                    "error": "Image to Visio conversion failed"
                }
            
            self.log(f"Generated Visio: {visio_path}", "SUCCESS")
            
        except ImportError:
            self.log("Visio conversion skipped", "WARNING")
            visio_path = None
        
        # Step 2: Upload to Tracker
        self.log("Step 2: Uploading to Alpha Tracker...")
        
        upload_results = []
        
        # Upload image
        image_upload = self.file_uploader.upload_file_to_project(
            project_number=project_number,
            file_path=str(image_path),
            file_type="plan_image_source"
        )
        upload_results.append(image_upload)
        
        # Upload Visio
        if visio_path:
            visio_upload = self.file_uploader.upload_file_to_project(
                project_number=project_number,
                file_path=str(visio_path),
                file_type="survey_plan_visio"
            )
            upload_results.append(visio_upload)
        
        return {
            "status": "success",
            "project_number": project_number,
            "image_path": str(image_path),
            "visio_path": str(visio_path) if visio_path else None,
            "conversion": conversion_result if visio_path else None,
            "uploads": upload_results
        }


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process survey plans from Outlook to Tracker"
    )
    parser.add_argument("--project", "-p", required=True, help="Project number")
    parser.add_argument("--email-subject", "-s", help="Email subject search term")
    parser.add_argument("--image", "-i", help="Direct image path (skips Outlook)")
    parser.add_argument("--output", "-o", help="Output folder for Visio")
    parser.add_argument("--client", "-c", default="cardtronics", help="Client type")
    
    args = parser.parse_args()
    
    pipeline = OutlookToTrackerPipeline(
        client_type=args.client,
        verbose=True
    )
    
    if args.image:
        # Process image file directly
        result = pipeline.process_plan_image(
            project_number=args.project,
            image_path=args.image,
            output_folder=args.output
        )
    else:
        # Process Outlook email
        result = pipeline.process_outlook_email(
            project_number=args.project,
            email_subject_hint=args.email_subject,
            output_folder=args.output
        )
    
    print(f"\nResult: {result['status']}")
    if result['status'] == 'success':
        print(f"  Image: {result.get('image_path')}")
        print(f"  Visio: {result.get('visio_path')}")
        print(f"  Uploads: {len(result.get('uploads', []))} files")
    else:
        print(f"  Error: {result.get('error')}")
