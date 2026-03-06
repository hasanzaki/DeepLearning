"""
Download all files from a Google Drive folder into a local directory.

Auth modes (in priority order):
  1. Service account JSON  (GOOGLE_SERVICE_ACCOUNT_JSON env var)
  2. Public folder via gdown  (no credentials needed if folder is public)

Usage:
    python ingestion/gdrive_downloader.py
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import GDRIVE_FOLDER_ID, GOOGLE_SERVICE_ACCOUNT_JSON

logger = logging.getLogger(__name__)

RAW_DOCS_DIR = Path(__file__).resolve().parents[1] / "ingestion" / "raw_docs"
SUPPORTED_MIME_TYPES = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/msword": ".doc",
    "text/plain": ".txt",
    "application/vnd.google-apps.document": ".docx",   # exported as DOCX
}
GDOCS_EXPORT_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


# ---------------------------------------------------------------------------
# Service-account downloader
# ---------------------------------------------------------------------------

def _build_service():
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    sa_path = Path(GOOGLE_SERVICE_ACCOUNT_JSON)
    if not sa_path.exists():
        raise FileNotFoundError(
            f"Service account JSON not found at {sa_path}. "
            "Set GOOGLE_SERVICE_ACCOUNT_JSON or use a public folder."
        )
    creds = service_account.Credentials.from_service_account_file(
        str(sa_path),
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    return build("drive", "v3", credentials=creds)


def _list_files_in_folder(service, folder_id: str) -> list[dict]:
    """Recursively list all supported files under folder_id."""
    results = []
    page_token = None
    query = f"'{folder_id}' in parents and trashed = false"

    while True:
        resp = service.files().list(
            q=query,
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token,
            pageSize=1000,
        ).execute()

        for item in resp.get("files", []):
            if item["mimeType"] == "application/vnd.google-apps.folder":
                # recurse into sub-folder
                results.extend(_list_files_in_folder(service, item["id"]))
            elif item["mimeType"] in SUPPORTED_MIME_TYPES:
                results.append(item)
            else:
                logger.debug("Skipping unsupported type: %s (%s)", item["name"], item["mimeType"])

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return results


def _download_file(service, file_meta: dict, dest_dir: Path) -> Optional[Path]:
    from googleapiclient.http import MediaIoBaseDownload
    import io

    file_id = file_meta["id"]
    name = file_meta["name"]
    mime = file_meta["mimeType"]
    ext = SUPPORTED_MIME_TYPES[mime]

    # Ensure correct extension on filename
    stem = Path(name).stem
    dest_path = dest_dir / f"{stem}{ext}"

    # Skip if already downloaded
    if dest_path.exists():
        logger.info("Already exists, skipping: %s", dest_path.name)
        return dest_path

    try:
        if mime == "application/vnd.google-apps.document":
            request = service.files().export_media(fileId=file_id, mimeType=GDOCS_EXPORT_MIME)
        else:
            request = service.files().get_media(fileId=file_id)

        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        dest_path.write_bytes(buffer.getvalue())
        logger.info("Downloaded: %s", dest_path.name)
        return dest_path

    except Exception as exc:
        logger.error("Failed to download %s: %s", name, exc)
        return None


def download_with_service_account(folder_id: str, dest_dir: Path) -> list[Path]:
    service = _build_service()
    files = _list_files_in_folder(service, folder_id)
    logger.info("Found %d supported files in Drive folder", len(files))

    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for f in files:
        path = _download_file(service, f, dest_dir)
        if path:
            downloaded.append(path)

    return downloaded


# ---------------------------------------------------------------------------
# Public-folder fallback via gdown
# ---------------------------------------------------------------------------

def download_public_folder(folder_id: str, dest_dir: Path) -> list[Path]:
    try:
        import gdown
    except ImportError:
        raise ImportError("Install gdown: pip install gdown")

    dest_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    logger.info("Downloading public folder via gdown: %s", url)
    gdown.download_folder(url=url, output=str(dest_dir), quiet=False, use_cookies=False)

    paths = list(dest_dir.rglob("*"))
    return [p for p in paths if p.is_file()]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def download(folder_id: Optional[str] = None, dest_dir: Optional[Path] = None) -> list[Path]:
    folder_id = folder_id or GDRIVE_FOLDER_ID
    dest_dir = dest_dir or RAW_DOCS_DIR

    if not folder_id:
        raise ValueError("GDRIVE_FOLDER_ID is not set. Add it to config/.env")

    sa_path = Path(GOOGLE_SERVICE_ACCOUNT_JSON)
    if sa_path.exists():
        logger.info("Using service account: %s", sa_path)
        return download_with_service_account(folder_id, dest_dir)
    else:
        logger.warning(
            "Service account JSON not found at %s. "
            "Falling back to gdown (requires folder to be publicly shared).",
            sa_path,
        )
        return download_public_folder(folder_id, dest_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    paths = download()
    print(f"\nDownloaded {len(paths)} files to {RAW_DOCS_DIR}")
    for p in paths:
        print(f"  {p.relative_to(RAW_DOCS_DIR.parent)}")
