#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================
"""
p12_download_preflight_GSE175634.py

External-validation acquisition/preflight for GDIS-Bio.

Dataset: GSE175634
Human iPSC -> cardiac differentiation scRNA-seq.

This script:
  1. downloads the uncorrected UMI count matrix and accompanying metadata,
  2. performs gzip/SHA-256 integrity checks,
  3. inspects cell metadata, cell/gene indices, and Matrix Market dimensions,
  4. checks whether deposited pseudotime, differentiation day, individual,
     and cell-type annotations are available.

It does NOT normalize data, reconstruct trajectories, compute PCA, or run GDIS.
"""

from pathlib import Path
import gzip
import hashlib
import shutil
import subprocess
import sys
import time

import pandas as pd

try:
    import requests
except ImportError as exc:
    raise SystemExit(
        "Install requests first:\n"
        "  python -m pip install requests"
    ) from exc


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR

DATASET_DIR = PROJECT_DIR / "raw_data" / "GSE175634"
RESULTS_DIR = PROJECT_DIR / "results" / "p12_external_GSE175634_download_preflight"
TABLE_DIR = RESULTS_DIR / "tables"
REPORT_FILE = RESULTS_DIR / "p12_external_GSE175634_download_preflight_report.txt"

BASE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/"
    "GSE175nnn/GSE175634/suppl/"
)

FILES = [
    "GSE175634_cell_metadata.tsv.gz",
    "GSE175634_cell_indices.tsv.gz",
    "GSE175634_gene_indices_counts.tsv.gz",
    "GSE175634_collection_metadata.txt.gz",
    "GSE175634_experimental_design.txt.gz",
    "GSE175634_cell_counts.mtx.gz",
]

DOWNLOAD_CHUNK = 8 * 1024 * 1024
HASH_CHUNK = 16 * 1024 * 1024
TIMEOUT = (30, 300)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def write_both(report, text=""):
    print(text)
    report.write(text + "\n")
    report.flush()


def human_bytes(value):
    value = float(value)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(HASH_CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def gzip_test(path):
    """Return (passed, message)."""
    gzip_exe = shutil.which("gzip")

    if gzip_exe:
        result = subprocess.run(
            [gzip_exe, "-t", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.returncode == 0, result.stderr.strip()

    try:
        with gzip.open(path, "rb") as handle:
            while handle.read(HASH_CHUNK):
                pass
        return True, ""
    except Exception as exc:
        return False, str(exc)


def get_remote_size(session, url):
    try:
        response = session.head(
            url,
            allow_redirects=True,
            timeout=TIMEOUT,
        )
        if response.status_code >= 400:
            return None
        length = response.headers.get("Content-Length")
        return int(length) if length else None
    except Exception:
        return None


def download_with_resume(session, url, destination):
    """Download with HTTP Range resume when supported."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    remote_size = get_remote_size(session, url)
    existing = destination.stat().st_size if destination.exists() else 0

    if remote_size and existing == remote_size:
        return "already_complete", remote_size

    if remote_size and existing > remote_size:
        destination.unlink()
        existing = 0

    headers = {}
    mode = "wb"

    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"

    response = session.get(
        url,
        headers=headers,
        stream=True,
        timeout=TIMEOUT,
        allow_redirects=True,
    )
    response.raise_for_status()

    # If server ignored Range, restart instead of appending a full file.
    if existing > 0 and response.status_code == 200:
        existing = 0
        mode = "wb"

    downloaded = existing
    start = time.time()
    last_print = start

    with open(destination, mode) as handle:
        for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK):
            if not chunk:
                continue

            handle.write(chunk)
            downloaded += len(chunk)

            now = time.time()
            if now - last_print >= 10:
                speed = (downloaded - existing) / max(now - start, 1e-9)

                if remote_size:
                    percent = 100.0 * downloaded / remote_size
                    print(
                        f"    {human_bytes(downloaded)} / "
                        f"{human_bytes(remote_size)} "
                        f"({percent:.1f}%) | {human_bytes(speed)}/s",
                        flush=True,
                    )
                else:
                    print(
                        f"    {human_bytes(downloaded)} | "
                        f"{human_bytes(speed)}/s",
                        flush=True,
                    )
                last_print = now

    local_size = destination.stat().st_size

    if remote_size and local_size != remote_size:
        raise RuntimeError(
            f"Download-size mismatch for {destination.name}: "
            f"local={local_size}, remote={remote_size}"
        )

    status = "resumed_and_completed" if existing > 0 else "downloaded"
    return status, remote_size


def read_matrix_market_header(path):
    """Read matrix dimensions without loading the sparse matrix."""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        header = handle.readline().strip()

        if not header.startswith("%%MatrixMarket"):
            raise ValueError("Not a Matrix Market file.")

        for line in handle:
            line = line.strip()

            if not line or line.startswith("%"):
                continue

            fields = line.split()

            if len(fields) < 3:
                raise ValueError("Could not parse Matrix Market dimensions.")

            return {
                "header": header,
                "rows": int(fields[0]),
                "cols": int(fields[1]),
                "nnz": int(fields[2]),
            }

    raise ValueError("Matrix dimension line not found.")


def read_tsv(path):
    return pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )


def read_small_text(path):
    try:
        df = pd.read_csv(
            path,
            sep="\t",
            compression="gzip",
            low_memory=False,
        )
        if df.shape[1] > 1:
            return df
    except Exception:
        pass

    return pd.read_csv(
        path,
        sep=None,
        engine="python",
        compression="gzip",
        low_memory=False,
    )


def preview_unique(series, limit=20):
    values = sorted(
        series.dropna().astype(str).unique().tolist()
    )
    if len(values) <= limit:
        return values
    return values[:limit] + [f"... ({len(values)} total)"]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {"User-Agent": "GDIS-Bio external validation"}
    )

    manifest_rows = []

    with open(REPORT_FILE, "w", encoding="utf-8") as report:

        write_both(report, "=" * 92)
        write_both(report, "GDIS-Bio External Validation Acquisition and Preflight")
        write_both(report, "Dataset: GSE175634")
        write_both(report, "=" * 92)

        write_both(
            report,
            "\nNo normalization, PCA, trajectory reconstruction, or GDIS "
            "is performed in p12.",
        )

        # -------------------------------------------------------------
        # 1. Download
        # -------------------------------------------------------------
        write_both(report, "\n1. DOWNLOAD AND INTEGRITY")
        write_both(report, "-" * 92)

        for filename in FILES:
            url = BASE_URL + filename
            destination = DATASET_DIR / filename

            write_both(report, f"\n{filename}")
            write_both(report, f"  Source: {url}")

            status, remote_size = download_with_resume(
                session,
                url,
                destination,
            )

            local_size = destination.stat().st_size

            write_both(report, f"  Status: {status}")
            write_both(report, f"  Local size: {human_bytes(local_size)}")

            gzip_ok, gzip_message = gzip_test(destination)

            write_both(
                report,
                f"  gzip integrity: {'PASS' if gzip_ok else 'FAIL'}",
            )

            if not gzip_ok:
                raise RuntimeError(
                    f"Gzip integrity failed for {filename}: {gzip_message}"
                )

            write_both(report, "  SHA-256 calculation...")
            checksum = sha256_file(destination)
            write_both(report, f"  SHA-256: {checksum}")

            manifest_rows.append(
                {
                    "filename": filename,
                    "url": url,
                    "status": status,
                    "remote_size_bytes": remote_size,
                    "local_size_bytes": local_size,
                    "gzip_integrity_pass": gzip_ok,
                    "sha256": checksum,
                }
            )

        manifest = pd.DataFrame(manifest_rows)
        manifest.to_csv(
            TABLE_DIR / "01_download_manifest.csv",
            index=False,
        )

        # -------------------------------------------------------------
        # 2. Metadata
        # -------------------------------------------------------------
        write_both(report, "\n2. CELL METADATA")
        write_both(report, "-" * 92)

        metadata = read_tsv(
            DATASET_DIR / "GSE175634_cell_metadata.tsv.gz"
        )

        write_both(
            report,
            f"Shape: {metadata.shape[0]:,} rows x "
            f"{metadata.shape[1]} columns",
        )

        write_both(report, "Columns:")
        for col in metadata.columns:
            write_both(report, f"  - {col}")

        required = [
            "cell",
            "exp.grp",
            "sample",
            "diffday",
            "individual",
            "type",
            "dpt_pseudotime",
        ]

        metadata_checks = {}

        write_both(report, "\nRequired external-validation fields:")
        for col in required:
            present = col in metadata.columns
            metadata_checks[col] = present
            write_both(
                report,
                f"  {'[PASS]' if present else '[MISSING]'} {col}",
            )

        for col in ["diffday", "individual", "type", "exp.grp", "sample"]:
            if col in metadata.columns:
                write_both(
                    report,
                    f"\n{col} values: "
                    f"{preview_unique(metadata[col])}",
                )

        if "dpt_pseudotime" in metadata.columns:
            pseudo = pd.to_numeric(
                metadata["dpt_pseudotime"],
                errors="coerce",
            )

            write_both(report, "\nDeposited diffusion pseudotime:")
            write_both(
                report,
                f"  finite/non-missing: {pseudo.notna().sum():,} / "
                f"{len(pseudo):,}",
            )

            if pseudo.notna().any():
                write_both(report, f"  min: {pseudo.min():.6f}")
                write_both(report, f"  median: {pseudo.median():.6f}")
                write_both(report, f"  max: {pseudo.max():.6f}")

        # -------------------------------------------------------------
        # 3. Indices and matrix header
        # -------------------------------------------------------------
        write_both(report, "\n3. CELL/GENE INDICES AND COUNT MATRIX")
        write_both(report, "-" * 92)

        cell_index = read_tsv(
            DATASET_DIR / "GSE175634_cell_indices.tsv.gz"
        )

        gene_index = read_tsv(
            DATASET_DIR / "GSE175634_gene_indices_counts.tsv.gz"
        )

        write_both(
            report,
            f"Cell-index shape: {cell_index.shape[0]:,} x "
            f"{cell_index.shape[1]}",
        )
        write_both(
            report,
            f"Cell-index columns: {list(cell_index.columns)}",
        )

        write_both(
            report,
            f"Gene-index shape: {gene_index.shape[0]:,} x "
            f"{gene_index.shape[1]}",
        )
        write_both(
            report,
            f"Gene-index columns: {list(gene_index.columns)}",
        )

        mm = read_matrix_market_header(
            DATASET_DIR / "GSE175634_cell_counts.mtx.gz"
        )

        write_both(report, f"Matrix header: {mm['header']}")
        write_both(
            report,
            f"Matrix dimensions: {mm['rows']:,} x {mm['cols']:,}",
        )
        write_both(
            report,
            f"Matrix reported nonzeros: {mm['nnz']:,}",
        )

        cells_x_genes = (
            mm["rows"] == len(cell_index)
            and mm["cols"] == len(gene_index)
        )

        genes_x_cells = (
            mm["rows"] == len(gene_index)
            and mm["cols"] == len(cell_index)
        )

        if cells_x_genes:
            orientation = "cells x genes"
            matrix_index_pass = True
        elif genes_x_cells:
            orientation = "genes x cells"
            matrix_index_pass = True
        else:
            orientation = "UNRESOLVED"
            matrix_index_pass = False

        write_both(report, f"Resolved orientation: {orientation}")
        write_both(
            report,
            f"Matrix/index dimensions: "
            f"{'[PASS]' if matrix_index_pass else '[REVIEW]'}",
        )

        # -------------------------------------------------------------
        # 4. Cell ID agreement
        # -------------------------------------------------------------
        write_both(report, "\n4. METADATA / CELL-INDEX ID AGREEMENT")
        write_both(report, "-" * 92)

        exact_id_pass = False
        best_column = None
        best_overlap = 0
        meta_unique = 0
        index_unique = 0

        if "cell" in metadata.columns:
            meta_ids = set(metadata["cell"].astype(str))
            meta_unique = len(meta_ids)

            for col in cell_index.columns:
                ids = set(cell_index[col].astype(str))
                overlap = len(meta_ids.intersection(ids))

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_column = col
                    index_unique = len(ids)

            exact_id_pass = (
                best_column is not None
                and best_overlap == meta_unique
                and best_overlap == index_unique
            )

            write_both(report, f"Best index-ID column: {best_column}")
            write_both(report, f"Metadata unique IDs: {meta_unique:,}")
            write_both(report, f"Index unique IDs: {index_unique:,}")
            write_both(report, f"Exact overlap: {best_overlap:,}")
            write_both(
                report,
                f"Exact ID agreement: "
                f"{'[PASS]' if exact_id_pass else '[REVIEW]'}",
            )

        # -------------------------------------------------------------
        # 5. Cell-type x differentiation-day inventory
        # -------------------------------------------------------------
        write_both(report, "\n5. CELL TYPE x DIFFERENTIATION DAY")
        write_both(report, "-" * 92)

        if "type" in metadata.columns and "diffday" in metadata.columns:
            cross = pd.crosstab(
                metadata["type"],
                metadata["diffday"],
                dropna=False,
            )

            cross.to_csv(
                TABLE_DIR / "02_cell_type_by_diffday_counts.csv"
            )

            write_both(report, cross.to_string())

        # -------------------------------------------------------------
        # 6. Individual counts
        # -------------------------------------------------------------
        write_both(report, "\n6. INDIVIDUAL / CELL-LINE INVENTORY")
        write_both(report, "-" * 92)

        if "individual" in metadata.columns:
            individual_counts = (
                metadata["individual"]
                .value_counts(dropna=False)
                .rename_axis("individual")
                .reset_index(name="n_cells")
            )

            individual_counts.to_csv(
                TABLE_DIR / "03_cells_by_individual.csv",
                index=False,
            )

            write_both(
                report,
                individual_counts.to_string(index=False),
            )

        # -------------------------------------------------------------
        # 7. Small experimental metadata
        # -------------------------------------------------------------
        write_both(report, "\n7. COLLECTION / EXPERIMENTAL DESIGN")
        write_both(report, "-" * 92)

        collection = read_small_text(
            DATASET_DIR / "GSE175634_collection_metadata.txt.gz"
        )

        design = read_small_text(
            DATASET_DIR / "GSE175634_experimental_design.txt.gz"
        )

        collection.to_csv(
            TABLE_DIR / "04_collection_metadata_parsed.csv",
            index=False,
        )

        design.to_csv(
            TABLE_DIR / "05_experimental_design_parsed.csv",
            index=False,
        )

        write_both(
            report,
            f"Collection metadata: "
            f"{collection.shape[0]:,} x {collection.shape[1]}",
        )
        write_both(
            report,
            f"Collection columns: {list(collection.columns)}",
        )

        write_both(
            report,
            f"Experimental design: "
            f"{design.shape[0]:,} x {design.shape[1]}",
        )
        write_both(
            report,
            f"Design columns: {list(design.columns)}",
        )

        # -------------------------------------------------------------
        # 8. Final qualification
        # -------------------------------------------------------------
        write_both(
            report,
            "\n8. PRELIMINARY EXTERNAL-VALIDATION QUALIFICATION",
        )
        write_both(report, "-" * 92)

        checks = [
            (
                "All six GEO files downloaded and passed gzip integrity",
                len(manifest) == len(FILES)
                and bool(manifest["gzip_integrity_pass"].all()),
            ),
            (
                "Metadata contains cell ID",
                metadata_checks.get("cell", False),
            ),
            (
                "Metadata contains differentiation day",
                metadata_checks.get("diffday", False),
            ),
            (
                "Metadata contains individual/cell-line identity",
                metadata_checks.get("individual", False),
            ),
            (
                "Metadata contains biological cell-type annotation",
                metadata_checks.get("type", False),
            ),
            (
                "Metadata contains deposited diffusion pseudotime",
                metadata_checks.get("dpt_pseudotime", False),
            ),
            (
                "Count-matrix dimensions match cell/gene indices",
                matrix_index_pass,
            ),
            (
                "Metadata IDs exactly match a cell-index column",
                exact_id_pass,
            ),
        ]

        passed = 0

        for label, status in checks:
            if status:
                passed += 1
                prefix = "[PASS]"
            else:
                prefix = "[REVIEW]"

            write_both(report, f"{prefix} {label}")

        write_both(
            report,
            f"\nPreliminary checks passed: {passed}/{len(checks)}",
        )

        if passed == len(checks):
            write_both(report, "\nFINAL STATUS: PASS")
            write_both(
                report,
                "GSE175634 is structurally ready for dedicated "
                "trajectory characterization.",
            )
        else:
            write_both(report, "\nFINAL STATUS: REVIEW REQUIRED")
            write_both(
                report,
                "Inspect the flagged metadata/index issue before "
                "trajectory characterization.",
            )

        write_both(
            report,
            "\nNext step: characterize cell types, differentiation days, "
            "individuals, and deposited diffusion pseudotime before "
            "defining the external-validation GDIS trajectories.",
        )

        write_both(report, f"\nRaw data: {DATASET_DIR}")
        write_both(report, f"Report:   {REPORT_FILE}")
        write_both(report, f"Tables:   {TABLE_DIR}")

    print("\n" + "=" * 92)
    print("p12_download_preflight_GSE175634.py completed.")
    print("=" * 92)
    print(f"Raw data: {DATASET_DIR}")
    print(f"Report  : {REPORT_FILE}")
    print(f"Tables  : {TABLE_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

