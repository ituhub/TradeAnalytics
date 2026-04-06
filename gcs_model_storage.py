"""
═══════════════════════════════════════════════════════════════════════════
  GCS Model Storage — Persist trained models on Google Cloud Storage
═══════════════════════════════════════════════════════════════════════════

  Solves:
  1. Cloud Run ephemeral filesystem — models lost on restart/redeploy
  2. Model sharing across instances — all containers see the same models
  3. Versioning — keeps track of when models were trained

  Usage:
    from gcs_model_storage import GCSModelStorage

    storage = GCSModelStorage()  # Auto-detects GCS credentials

    # After training:
    storage.upload_models(ticker)  # Uploads ./models/{ticker}_*.pt/.pkl to GCS

    # On startup / before backtest:
    storage.download_models(ticker)  # Downloads from GCS to ./models/

    # List what's stored:
    storage.list_models()  # Returns dict of tickers and their model files

    # Check freshness:
    info = storage.get_model_info(ticker)  # Returns timestamps, file sizes
═══════════════════════════════════════════════════════════════════════════
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════
# GCS AVAILABILITY CHECK
# ════════════════════════════════════════════════════════════════
GCS_AVAILABLE = False
try:
    from google.cloud import storage as gcs_storage
    GCS_AVAILABLE = True
except ImportError:
    logger.info("google-cloud-storage not installed — GCS model persistence disabled")

# ════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════
# Set via environment variable or default
GCS_BUCKET_NAME = os.getenv("GCS_MODEL_BUCKET", "ai-trading-pro-models")
GCS_PREFIX = os.getenv("GCS_MODEL_PREFIX", "trained_models/")
LOCAL_MODEL_DIR = Path("models")


class GCSModelStorage:
    """
    Handles uploading/downloading trained models to/from Google Cloud Storage.
    Falls back gracefully when GCS is not available (local-only mode).
    """

    def __init__(self, bucket_name: str = None, prefix: str = None):
        self.bucket_name = bucket_name or GCS_BUCKET_NAME
        self.prefix = prefix or GCS_PREFIX
        self.local_dir = LOCAL_MODEL_DIR
        self.client = None
        self.bucket = None
        self.available = False

        if GCS_AVAILABLE:
            try:
                self.client = gcs_storage.Client()
                self.bucket = self.client.bucket(self.bucket_name)
                # Verify bucket exists (or create it)
                if not self.bucket.exists():
                    logger.info(f"Creating GCS bucket: {self.bucket_name}")
                    self.bucket = self.client.create_bucket(self.bucket_name)
                self.available = True
                logger.info(f"✅ GCS model storage connected: gs://{self.bucket_name}/{self.prefix}")
            except Exception as e:
                logger.warning(f"GCS initialization failed: {e}")
                logger.info("Running in local-only mode — models saved to ./models/ only")
        else:
            logger.info("GCS not available — models saved to ./models/ only")

    # ────────────────────────────────────────────────────────────
    # UPLOAD: Local → GCS
    # ────────────────────────────────────────────────────────────
    def upload_models(self, ticker: str) -> Dict:
        """
        Upload all model files for a ticker from local ./models/ to GCS.
        Returns dict with upload results.
        """
        results = {
            'ticker': ticker,
            'uploaded': [],
            'failed': [],
            'gcs_available': self.available,
            'timestamp': datetime.now().isoformat(),
        }

        if not self.available:
            logger.info(f"GCS not available — models for {ticker} saved locally only")
            return results

        safe_ticker = self._safe_ticker(ticker)
        local_files = self._find_local_files(safe_ticker)

        if not local_files:
            logger.warning(f"No local model files found for {ticker}")
            return results

        for local_path in local_files:
            try:
                gcs_key = f"{self.prefix}{safe_ticker}/{local_path.name}"
                blob = self.bucket.blob(gcs_key)
                blob.upload_from_filename(str(local_path))

                file_size = local_path.stat().st_size
                results['uploaded'].append({
                    'file': local_path.name,
                    'gcs_path': f"gs://{self.bucket_name}/{gcs_key}",
                    'size_bytes': file_size,
                    'size_mb': round(file_size / (1024 * 1024), 2),
                })
                logger.info(f"✅ Uploaded {local_path.name} → gs://{self.bucket_name}/{gcs_key}")
            except Exception as e:
                results['failed'].append({'file': local_path.name, 'error': str(e)})
                logger.error(f"❌ Failed to upload {local_path.name}: {e}")

        total = len(results['uploaded'])
        logger.info(f"📤 Uploaded {total} model files for {ticker} to GCS")
        return results

    # ────────────────────────────────────────────────────────────
    # DOWNLOAD: GCS → Local
    # ────────────────────────────────────────────────────────────
    def download_models(self, ticker: str, force: bool = False) -> Dict:
        """
        Download all model files for a ticker from GCS to local ./models/.
        Skips files that already exist locally unless force=True.
        Returns dict with download results.
        """
        results = {
            'ticker': ticker,
            'downloaded': [],
            'skipped': [],
            'failed': [],
            'gcs_available': self.available,
            'timestamp': datetime.now().isoformat(),
        }

        if not self.available:
            logger.info(f"GCS not available — using local models only for {ticker}")
            return results

        safe_ticker = self._safe_ticker(ticker)
        gcs_prefix = f"{self.prefix}{safe_ticker}/"

        self.local_dir.mkdir(exist_ok=True)

        try:
            blobs = list(self.bucket.list_blobs(prefix=gcs_prefix))

            if not blobs:
                logger.info(f"No model files found on GCS for {ticker}")
                return results

            for blob in blobs:
                filename = blob.name.split('/')[-1]
                if not filename:
                    continue

                local_path = self.local_dir / filename

                # Skip if exists locally and not forcing
                if local_path.exists() and not force:
                    local_mtime = datetime.fromtimestamp(local_path.stat().st_mtime)
                    gcs_mtime = blob.updated

                    # Only skip if local is newer
                    if gcs_mtime and local_mtime.timestamp() >= gcs_mtime.timestamp():
                        results['skipped'].append({
                            'file': filename,
                            'reason': 'local is newer or same age',
                        })
                        continue

                try:
                    blob.download_to_filename(str(local_path))
                    results['downloaded'].append({
                        'file': filename,
                        'size_bytes': blob.size,
                        'gcs_updated': str(blob.updated),
                    })
                    logger.info(f"✅ Downloaded {filename} from GCS")
                except Exception as e:
                    results['failed'].append({'file': filename, 'error': str(e)})
                    logger.error(f"❌ Failed to download {filename}: {e}")

        except Exception as e:
            logger.error(f"GCS download error for {ticker}: {e}")
            results['failed'].append({'error': str(e)})

        total = len(results['downloaded'])
        logger.info(f"📥 Downloaded {total} model files for {ticker} from GCS")
        return results

    # ────────────────────────────────────────────────────────────
    # LIST: What's stored in GCS
    # ────────────────────────────────────────────────────────────
    def list_models(self) -> Dict[str, List[Dict]]:
        """
        List all model files stored in GCS, grouped by ticker.
        """
        if not self.available:
            # List local files instead
            return self._list_local_models()

        result = {}
        try:
            blobs = list(self.bucket.list_blobs(prefix=self.prefix))
            for blob in blobs:
                parts = blob.name.replace(self.prefix, '').split('/')
                if len(parts) >= 2 and parts[1]:
                    ticker = parts[0]
                    filename = parts[1]
                    if ticker not in result:
                        result[ticker] = []
                    result[ticker].append({
                        'file': filename,
                        'size_bytes': blob.size,
                        'size_mb': round((blob.size or 0) / (1024 * 1024), 2),
                        'updated': str(blob.updated) if blob.updated else 'unknown',
                        'gcs_path': f"gs://{self.bucket_name}/{blob.name}",
                    })
        except Exception as e:
            logger.error(f"GCS list error: {e}")

        return result

    # ────────────────────────────────────────────────────────────
    # MODEL INFO: Timestamps, sizes, freshness
    # ────────────────────────────────────────────────────────────
    def get_model_info(self, ticker: str) -> Dict:
        """
        Get detailed info about models for a ticker — both local and GCS.
        """
        safe_ticker = self._safe_ticker(ticker)
        info = {
            'ticker': ticker,
            'local': {'exists': False, 'files': [], 'total_size_mb': 0},
            'gcs': {'exists': False, 'files': [], 'total_size_mb': 0},
        }

        # Local info
        local_files = self._find_local_files(safe_ticker)
        if local_files:
            info['local']['exists'] = True
            total_size = 0
            for f in local_files:
                size = f.stat().st_size
                total_size += size
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                info['local']['files'].append({
                    'file': f.name,
                    'size_mb': round(size / (1024 * 1024), 2),
                    'modified': mtime.strftime('%Y-%m-%d %H:%M:%S'),
                })
            info['local']['total_size_mb'] = round(total_size / (1024 * 1024), 2)

        # GCS info
        if self.available:
            try:
                gcs_prefix = f"{self.prefix}{safe_ticker}/"
                blobs = list(self.bucket.list_blobs(prefix=gcs_prefix))
                if blobs:
                    info['gcs']['exists'] = True
                    total_size = 0
                    for blob in blobs:
                        filename = blob.name.split('/')[-1]
                        if filename:
                            total_size += (blob.size or 0)
                            info['gcs']['files'].append({
                                'file': filename,
                                'size_mb': round((blob.size or 0) / (1024 * 1024), 2),
                                'updated': str(blob.updated) if blob.updated else 'unknown',
                            })
                    info['gcs']['total_size_mb'] = round(total_size / (1024 * 1024), 2)
            except Exception as e:
                logger.warning(f"Could not get GCS info for {ticker}: {e}")

        return info

    # ────────────────────────────────────────────────────────────
    # SYNC: Ensure local and GCS are consistent
    # ────────────────────────────────────────────────────────────
    def sync_models(self, ticker: str) -> Dict:
        """
        Bidirectional sync: upload local files missing from GCS,
        download GCS files missing locally.
        """
        if not self.available:
            return {'status': 'gcs_not_available', 'ticker': ticker}

        safe_ticker = self._safe_ticker(ticker)

        # Get local and GCS file lists
        local_files = {f.name for f in self._find_local_files(safe_ticker)}
        gcs_files = set()
        try:
            gcs_prefix = f"{self.prefix}{safe_ticker}/"
            blobs = list(self.bucket.list_blobs(prefix=gcs_prefix))
            gcs_files = {blob.name.split('/')[-1] for blob in blobs if blob.name.split('/')[-1]}
        except Exception:
            pass

        # Upload files that are local but not in GCS
        upload_needed = local_files - gcs_files
        # Download files that are in GCS but not local
        download_needed = gcs_files - local_files

        result = {
            'ticker': ticker,
            'uploaded': [],
            'downloaded': [],
            'already_synced': len(local_files & gcs_files),
        }

        for filename in upload_needed:
            try:
                local_path = self.local_dir / filename
                gcs_key = f"{self.prefix}{safe_ticker}/{filename}"
                blob = self.bucket.blob(gcs_key)
                blob.upload_from_filename(str(local_path))
                result['uploaded'].append(filename)
                logger.info(f"📤 Synced {filename} → GCS")
            except Exception as e:
                logger.warning(f"Sync upload failed for {filename}: {e}")

        for filename in download_needed:
            try:
                gcs_key = f"{self.prefix}{safe_ticker}/{filename}"
                blob = self.bucket.blob(gcs_key)
                local_path = self.local_dir / filename
                blob.download_to_filename(str(local_path))
                result['downloaded'].append(filename)
                logger.info(f"📥 Synced {filename} ← GCS")
            except Exception as e:
                logger.warning(f"Sync download failed for {filename}: {e}")

        return result

    # ────────────────────────────────────────────────────────────
    # DELETE: Remove models from GCS
    # ────────────────────────────────────────────────────────────
    def delete_models(self, ticker: str) -> Dict:
        """Delete all model files for a ticker from GCS."""
        if not self.available:
            return {'status': 'gcs_not_available'}

        safe_ticker = self._safe_ticker(ticker)
        deleted = []

        try:
            gcs_prefix = f"{self.prefix}{safe_ticker}/"
            blobs = list(self.bucket.list_blobs(prefix=gcs_prefix))
            for blob in blobs:
                blob.delete()
                deleted.append(blob.name.split('/')[-1])
                logger.info(f"🗑️ Deleted {blob.name} from GCS")
        except Exception as e:
            logger.error(f"GCS delete error: {e}")

        return {'ticker': ticker, 'deleted': deleted}

    # ────────────────────────────────────────────────────────────
    # HELPERS
    # ────────────────────────────────────────────────────────────
    @staticmethod
    def _safe_ticker(ticker: str) -> str:
        """Convert ticker to safe filename."""
        return ticker.replace('/', '_').replace(' ', '_')

    def _find_local_files(self, safe_ticker: str) -> List[Path]:
        """Find all local model files for a ticker."""
        if not self.local_dir.exists():
            return []
        return sorted([
            f for f in self.local_dir.iterdir()
            if f.name.startswith(f"{safe_ticker}_")
            and f.suffix in ('.pt', '.pkl', '.h5', '.keras')
        ])

    def _list_local_models(self) -> Dict[str, List[Dict]]:
        """List local model files grouped by ticker."""
        result = {}
        if not self.local_dir.exists():
            return result

        for f in self.local_dir.iterdir():
            if f.suffix in ('.pt', '.pkl', '.h5', '.keras'):
                # Extract ticker from filename (everything before the last _modelname.ext)
                parts = f.stem.rsplit('_', 1)
                if len(parts) >= 1:
                    ticker = parts[0]
                    if ticker not in result:
                        result[ticker] = []
                    result[ticker].append({
                        'file': f.name,
                        'size_mb': round(f.stat().st_size / (1024 * 1024), 2),
                        'modified': datetime.fromtimestamp(f.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                    })
        return result
