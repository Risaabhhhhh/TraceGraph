"""
Download / verify the Elliptic Bitcoin dataset.

The Elliptic dataset is hosted on Kaggle and requires either:
  1. The Kaggle CLI (`pip install kaggle`) with valid API credentials, or
  2. Manual download from https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

This script will:
  - Attempt automated download via the Kaggle API.
  - If that fails, print clear manual-download instructions and exit.
  - Verify file integrity via SHA-256 checksums after download.
  - Extract the CSV files into ml/data/raw/.
"""

import hashlib
import os
import sys
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

KAGGLE_DATASET = "ellipticco/elliptic-data-set"

EXPECTED_FILES = {
    "elliptic_txs_features.csv",
    "elliptic_txs_classes.csv",
    "elliptic_txs_edgelist.csv",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_files(raw_dir: Path) -> dict[str, str]:
    """Check that all expected CSVs exist, compute and return checksums."""
    checksums = {}
    missing = []
    for fname in sorted(EXPECTED_FILES):
        fpath = raw_dir / fname
        if not fpath.exists():
            missing.append(fname)
        else:
            checksums[fname] = sha256_file(fpath)
    if missing:
        print(f"ERROR: Missing files in {raw_dir}:")
        for m in missing:
            print(f"  - {m}")
        print()
        print_manual_instructions()
        sys.exit(1)
    return checksums


def print_manual_instructions():
    """Print step-by-step manual download instructions."""
    print("=" * 72)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 72)
    print()
    print("The Elliptic Bitcoin dataset requires a Kaggle account.")
    print()
    print("Option A — Kaggle CLI (recommended):")
    print("  1. pip install kaggle")
    print("  2. Create API token: https://www.kaggle.com/settings -> API -> Create New Token")
    print("  3. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or")
    print("     C:\\Users\\<you>\\.kaggle\\ (Windows)")
    print(f"  4. Run: kaggle datasets download -d {KAGGLE_DATASET} -p {RAW_DIR} --unzip")
    print()
    print("Option B — Browser download:")
    print(f"  1. Go to https://www.kaggle.com/datasets/{KAGGLE_DATASET}")
    print("  2. Click 'Download' (requires sign-in).")
    print(f"  3. Extract the ZIP so that these files are directly in {RAW_DIR}/:")
    for fname in sorted(EXPECTED_FILES):
        print(f"       - {fname}")
    print()
    print("After downloading, re-run this script to verify checksums.")
    print("=" * 72)


def try_kaggle_download(raw_dir: Path) -> bool:
    """Attempt automated download via the Kaggle API. Returns True on success."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
    except ImportError:
        print("Kaggle API not installed (pip install kaggle). Trying manual path...")
        return False

    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print(f"Kaggle authentication failed: {e}")
        return False

    print(f"Downloading {KAGGLE_DATASET} via Kaggle API...")
    try:
        api.dataset_download_files(KAGGLE_DATASET, path=str(raw_dir), unzip=False)
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        return False

    # Find and extract the zip
    zip_files = list(raw_dir.glob("*.zip"))
    if not zip_files:
        print("Download completed but no ZIP found.")
        return False

    for zf in zip_files:
        print(f"Extracting {zf.name}...")
        with zipfile.ZipFile(zf, "r") as z:
            # Extract CSVs — they may be nested in a subdirectory inside the zip
            for member in z.namelist():
                basename = Path(member).name
                if basename in EXPECTED_FILES:
                    # Extract to raw_dir directly, flattening any subdirectory
                    target = raw_dir / basename
                    with z.open(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())
        zf.unlink()  # remove ZIP after extraction

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    raw_dir = RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Check if files already present
    all_present = all((raw_dir / f).exists() for f in EXPECTED_FILES)

    if not all_present:
        success = try_kaggle_download(raw_dir)
        if not success:
            # Check again — maybe user placed them manually
            all_present = all((raw_dir / f).exists() for f in EXPECTED_FILES)
            if not all_present:
                print_manual_instructions()
                sys.exit(1)

    # Verify
    checksums = verify_files(raw_dir)

    print()
    print("Elliptic dataset verified. SHA-256 checksums:")
    for fname, digest in sorted(checksums.items()):
        print(f"  {fname}: {digest}")

    # Write checksums to a sidecar file for DATA.md reference
    checksum_path = raw_dir / "elliptic_checksums.txt"
    with open(checksum_path, "w") as f:
        for fname, digest in sorted(checksums.items()):
            f.write(f"{digest}  {fname}\n")
    print(f"\nChecksums written to {checksum_path}")

    return checksums


if __name__ == "__main__":
    main()
