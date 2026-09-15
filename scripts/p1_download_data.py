#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================
"""
p1_download_data.py

Download the essential GSE114412 Stage-5 files for the GDIS-Bio project.

The script downloads:
1. Processed gene-expression counts
2. Complete Stage-5 cell metadata
3. Endocrine pseudotime metadata

No third-party Python packages are required.
"""

from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import shutil
import sys


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Directory where the downloaded files will be stored.
DOWNLOAD_DIR = Path("raw_data")

# GEO supplementary-file base URL.
BASE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/"
    "GSE114nnn/GSE114412/suppl/"
)

# Files required for the initial GDIS-Bio dataset preflight.
FILES = [
    "GSE114412_Stage_5.all.processed_counts.tsv.gz",
    "GSE114412_Stage_5.all.cell_metadata.tsv.gz",
    "GSE114412_Stage_5.endocrine_pseudotime.cell_metadata.tsv.gz",
]


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def human_readable_size(num_bytes):
    """Convert bytes to a human-readable file size."""
    size = float(num_bytes)

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0

    return f"{size:.2f} PB"


def download_file(url, destination):
    """
    Download one file from URL to destination.

    A temporary '.part' file is used so an interrupted download is not
    mistaken for a complete dataset file.
    """
    temp_file = destination.with_suffix(destination.suffix + ".part")

    print(f"\nDownloading:")
    print(f"  URL : {url}")
    print(f"  File: {destination}")

    # Set a User-Agent because some web servers reject requests without one.
    request = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    try:
        with urlopen(request) as response, open(temp_file, "wb") as out_file:
            total_size = response.headers.get("Content-Length")

            if total_size is not None:
                print(
                    f"  Expected size: "
                    f"{human_readable_size(int(total_size))}"
                )

            # Copy data in 1 MB blocks.
            shutil.copyfileobj(response, out_file, length=1024 * 1024)

        # Rename only after the download completes successfully.
        temp_file.replace(destination)

        print(
            f"  Completed: "
            f"{human_readable_size(destination.stat().st_size)}"
        )

        return True

    except HTTPError as error:
        print(f"  HTTP error {error.code}: {error.reason}")

    except URLError as error:
        print(f"  URL/network error: {error.reason}")

    except KeyboardInterrupt:
        print("\n  Download interrupted by user.")

    except Exception as error:
        print(f"  Unexpected error: {error}")

    # Remove an incomplete temporary file after a failed download.
    if temp_file.exists():
        temp_file.unlink()

    return False


# ---------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------

def main():
    """Download all required GSE114412 Stage-5 files."""

    print("=" * 70)
    print("GDIS-Bio Dataset Download")
    print("Dataset: GSE114412 - Stage 5 pancreatic differentiation")
    print("=" * 70)

    # Create the output directory if it does not already exist.
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nDownload directory: {DOWNLOAD_DIR.resolve()}")

    successful = 0

    for filename in FILES:
        destination = DOWNLOAD_DIR / filename
        url = BASE_URL + filename

        # Do not download a file again if it already exists and is non-empty.
        if destination.exists() and destination.stat().st_size > 0:
            print(f"\nAlready present, skipping:")
            print(f"  {destination}")
            print(
                f"  Size: "
                f"{human_readable_size(destination.stat().st_size)}"
            )
            successful += 1
            continue

        if download_file(url, destination):
            successful += 1

    # -----------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------

    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)

    for filename in FILES:
        path = DOWNLOAD_DIR / filename

        if path.exists():
            print(
                f"[OK]     {filename:<62} "
                f"{human_readable_size(path.stat().st_size)}"
            )
        else:
            print(f"[MISSING] {filename}")

    print()
    print(f"Successfully available: {successful}/{len(FILES)} files")

    if successful == len(FILES):
        print("\nAll required files are ready for the GDIS preflight analysis.")
        return 0

    print("\nOne or more files could not be downloaded.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

