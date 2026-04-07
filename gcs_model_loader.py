"""
GCS Model Loader — Download Colab-trained models on Cloud Run startup
=====================================================================
Integrates with enhprog.py's load_trained_models() function.

Models are stored flat in GCS:
  gs://ai-trading-models-91e8/models/BTCUSD_cnn_lstm.pt
  gs://ai-trading-models-91e8/models/BTCUSD_scaler.pkl
  gs://ai-trading-models-91e8/models/BTCUSD_config.pkl

This module downloads them to local ./models/ so load_trained_models() works.

Usage (in enhprog.py or app.py):
    from gcs_model_loader import ensure_models_available
    ensure_models_available()  # Downloads all models from GCS on first call
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Configuration from environment
GCS_BUCKET = os.getenv("GCS_MODEL_BUCKET", os.getenv("GCS_BUCKET_NAME", "ai-trading-models-91e8"))
GCS_PREFIX = os.getenv("GCS_MODEL_PREFIX", "models/")
LOCAL_MODEL_DIR = Path("models")

# Track if we've already downloaded
_models_downloaded = False


def ensure_models_available(tickers=None, force=False):
    """
    Download all models from GCS to local ./models/ directory.
    Called once on startup. Skips if already downloaded unless force=True.
    """
    global _models_downloaded

    if _models_downloaded and not force:
        return True

    try:
        from google.cloud import storage as gcs_storage
    except ImportError:
        logger.info("google-cloud-storage not installed — skipping GCS model download")
        return False

    try:
        project = os.getenv("GOOGLE_CLOUD_PROJECT", "")
        if project:
            client = gcs_storage.Client(project=project)
        else:
            client = gcs_storage.Client()

        bucket = client.bucket(GCS_BUCKET)

        if not bucket.exists():
            logger.warning(f"GCS bucket '{GCS_BUCKET}' not found")
            return False

        # Create local model directory
        LOCAL_MODEL_DIR.mkdir(exist_ok=True)

        # List all model files in the bucket
        blobs = list(bucket.list_blobs(prefix=GCS_PREFIX))

        if not blobs:
            logger.info(f"No model files found in gs://{GCS_BUCKET}/{GCS_PREFIX}")
            return False

        # Filter by tickers if specified
        if tickers:
            from enhprog import safe_ticker_name
            safe_tickers = {safe_ticker_name(t) for t in tickers}
            blobs = [b for b in blobs if any(
                b.name.replace(GCS_PREFIX, '').startswith(st) for st in safe_tickers
            )]

        downloaded = 0
        skipped = 0

        for blob in blobs:
            # Extract filename from the blob path
            filename = blob.name.replace(GCS_PREFIX, '').strip('/')

            # Skip directories / empty names
            if not filename or filename.endswith('/'):
                continue

            # Handle nested paths (ticker/model.pt) → flatten to models/ticker_model.pt
            # Our Colab script uses flat: models/BTCUSD_cnn_lstm.pt
            # So filename is already correct
            local_path = LOCAL_MODEL_DIR / filename

            # Skip if file already exists and is same size
            if local_path.exists() and not force:
                local_size = local_path.stat().st_size
                if blob.size and local_size == blob.size:
                    skipped += 1
                    continue

            try:
                blob.download_to_filename(str(local_path))
                downloaded += 1
            except Exception as e:
                logger.warning(f"Failed to download {filename}: {e}")

        _models_downloaded = True
        logger.info(f"✅ GCS model sync: {downloaded} downloaded, {skipped} skipped "
                    f"(from gs://{GCS_BUCKET}/{GCS_PREFIX})")
        return True

    except Exception as e:
        logger.warning(f"GCS model download failed: {e}")
        return False


def download_ticker_models(ticker):
    """
    Download models for a specific ticker from GCS.
    Called by load_trained_models() if local files don't exist.
    """
    try:
        from google.cloud import storage as gcs_storage
    except ImportError:
        return False

    try:
        from enhprog import safe_ticker_name
        safe_ticker = safe_ticker_name(ticker)
    except ImportError:
        safe_ticker = ticker.replace('/', '_').replace(' ', '_')

    try:
        project = os.getenv("GOOGLE_CLOUD_PROJECT", "")
        client = gcs_storage.Client(project=project) if project else gcs_storage.Client()
        bucket = client.bucket(GCS_BUCKET)

        LOCAL_MODEL_DIR.mkdir(exist_ok=True)

        # List files for this ticker
        blobs = list(bucket.list_blobs(prefix=f"{GCS_PREFIX}{safe_ticker}_"))

        if not blobs:
            logger.info(f"No GCS models found for {ticker}")
            return False

        downloaded = 0
        for blob in blobs:
            filename = blob.name.replace(GCS_PREFIX, '').strip('/')
            if not filename:
                continue

            local_path = LOCAL_MODEL_DIR / filename
            try:
                blob.download_to_filename(str(local_path))
                downloaded += 1
            except Exception as e:
                logger.warning(f"Failed to download {filename}: {e}")

        logger.info(f"✅ Downloaded {downloaded} model files for {ticker} from GCS")
        return downloaded > 0

    except Exception as e:
        logger.warning(f"GCS download for {ticker} failed: {e}")
        return False
