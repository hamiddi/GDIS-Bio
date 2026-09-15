#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================
"""
p2_preflight_data.py

Corrected preflight inspection of GSE114412 Stage-5 data for GDIS-Bio.

IMPORTANT
---------
The processed-count matrix is organized as:

    rows    = cells
    columns = genes

The first column is '# library.barcode'.

This program performs inspection only. It does NOT normalize, filter,
transform, run PCA, infer trajectories, or calculate GDIS.
"""

from pathlib import Path
import sys
import gzip
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

if SCRIPT_DIR.name == "scripts":
    PROJECT_DIR = SCRIPT_DIR.parent
else:
    PROJECT_DIR = SCRIPT_DIR

RAW_DIR = PROJECT_DIR / "raw_data"
RESULTS_DIR = PROJECT_DIR / "results"

COUNTS_FILE = RAW_DIR / "GSE114412_Stage_5.all.processed_counts.tsv.gz"
METADATA_FILE = RAW_DIR / "GSE114412_Stage_5.all.cell_metadata.tsv.gz"
PSEUDOTIME_FILE = (
    RAW_DIR / "GSE114412_Stage_5.endocrine_pseudotime.cell_metadata.tsv.gz"
)

REPORT_FILE = RESULTS_DIR / "p2_preflight_report_v2.txt"

MARKER_GENES = [
    "NEUROG3",
    "PDX1",
    "NKX6-1",
    "INS",
    "PAX4",
    "NEUROD1",
    "TPH1",
    "SLC18A1",
    "ISL1",
    "PTF1A",
]


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def human_readable_size(num_bytes):
    """Convert a byte count into a human-readable string."""
    size = float(num_bytes)

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0

    return f"{size:.2f} PB"


def write_both(report, text=""):
    """Print to screen and write the same text to the report."""
    print(text)
    report.write(text + "\n")


def find_candidate_columns(columns, keywords):
    """Find columns containing any of the supplied keywords."""
    matches = []

    for col in columns:
        low = str(col).lower()

        if any(keyword.lower() in low for keyword in keywords):
            matches.append(col)

    return matches


def normalize_ids(values):
    """Normalize identifiers for reliable comparisons."""
    return (
        pd.Series(values, dtype="string")
        .str.strip()
        .dropna()
    )


def count_gzip_rows(path):
    """Count data rows in a gzipped TSV, excluding the header."""
    n_rows = 0

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        next(fh)

        for _ in fh:
            n_rows += 1

    return n_rows


# ---------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------

def main():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(REPORT_FILE, "w", encoding="utf-8") as report:

        write_both(report, "=" * 78)
        write_both(report, "GDIS-Bio Corrected Preflight Report")
        write_both(report, "Dataset: GSE114412 Stage 5")
        write_both(report, "=" * 78)

        # -------------------------------------------------------------
        # 1. File checks
        # -------------------------------------------------------------

        write_both(report, "\n1. FILE CHECKS")
        write_both(report, "-" * 78)

        required_files = [
            ("Processed counts", COUNTS_FILE),
            ("Cell metadata", METADATA_FILE),
            ("Endocrine pseudotime metadata", PSEUDOTIME_FILE),
        ]

        missing = False

        for label, path in required_files:
            if path.exists() and path.stat().st_size > 0:
                write_both(
                    report,
                    f"[OK] {label}: {path.name} "
                    f"({human_readable_size(path.stat().st_size)})"
                )
            else:
                write_both(report, f"[MISSING] {label}: {path}")
                missing = True

        if missing:
            write_both(report, "\nERROR: Required files are missing.")
            return 1

        # -------------------------------------------------------------
        # 2. Main metadata
        # -------------------------------------------------------------

        write_both(report, "\n2. MAIN CELL METADATA")
        write_both(report, "-" * 78)

        metadata = pd.read_csv(
            METADATA_FILE,
            sep="\t",
            compression="gzip",
            low_memory=False,
        )

        write_both(
            report,
            f"Dimensions: {metadata.shape[0]:,} rows x "
            f"{metadata.shape[1]:,} columns"
        )

        write_both(report, "\nColumns:")

        for i, col in enumerate(metadata.columns, start=1):
            write_both(report, f"  {i:2d}. {col}")

        time_cols = find_candidate_columns(
            metadata.columns,
            ["day", "time", "week", "stage"]
        )

        replicate_cols = find_candidate_columns(
            metadata.columns,
            ["diff", "rep", "batch", "experiment", "sample"]
        )

        celltype_cols = find_candidate_columns(
            metadata.columns,
            ["cluster", "celltype", "cell_type", "annotation", "identity"]
        )

        write_both(
            report,
            "\nCandidate temporal columns: "
            + (", ".join(time_cols) if time_cols else "None")
        )

        write_both(
            report,
            "Candidate replicate/sample columns: "
            + (", ".join(replicate_cols) if replicate_cols else "None")
        )

        write_both(
            report,
            "Candidate cell-type columns: "
            + (", ".join(celltype_cols) if celltype_cols else "None")
        )

        # Print distributions.
        for col in dict.fromkeys(time_cols + replicate_cols + celltype_cols):
            nunique = metadata[col].nunique(dropna=False)

            if nunique <= 100:
                write_both(
                    report,
                    f"\nValue counts for '{col}' "
                    f"({nunique} unique values):"
                )

                vc = metadata[col].value_counts(dropna=False)

                for value, count in vc.items():
                    write_both(
                        report,
                        f"  {str(value):40s} {count:8,d}"
                    )

        # -------------------------------------------------------------
        # 3. Count matrix orientation and dimensions
        # -------------------------------------------------------------

        write_both(report, "\n3. PROCESSED EXPRESSION MATRIX")
        write_both(report, "-" * 78)

        # Read header only.
        header = pd.read_csv(
            COUNTS_FILE,
            sep="\t",
            compression="gzip",
            nrows=0,
        )

        columns = list(header.columns)

        id_column = columns[0]
        gene_columns = columns[1:]

        n_cells = count_gzip_rows(COUNTS_FILE)
        n_genes = len(gene_columns)

        write_both(report, f"First column: {id_column}")
        write_both(report, "Matrix orientation: cells x genes")
        write_both(report, f"Cells: {n_cells:,}")
        write_both(report, f"Genes: {n_genes:,}")

        write_both(report, "\nFirst 10 gene columns:")

        for gene in gene_columns[:10]:
            write_both(report, f"  {gene}")

        # -------------------------------------------------------------
        # 4. Cell-ID matching
        # -------------------------------------------------------------

        write_both(report, "\n4. CELL-ID MATCHING")
        write_both(report, "-" * 78)

        count_ids_df = pd.read_csv(
            COUNTS_FILE,
            sep="\t",
            compression="gzip",
            usecols=[0],
            low_memory=False,
        )

        count_ids = normalize_ids(count_ids_df.iloc[:, 0])
        count_id_set = set(count_ids)

        metadata_id_col = "library.barcode"

        if metadata_id_col not in metadata.columns:
            write_both(
                report,
                f"[FAIL] '{metadata_id_col}' was not found in metadata."
            )
            metadata_id_set = set()
        else:
            metadata_ids = normalize_ids(metadata[metadata_id_col])
            metadata_id_set = set(metadata_ids)

            overlap = count_id_set & metadata_id_set
            only_counts = count_id_set - metadata_id_set
            only_metadata = metadata_id_set - count_id_set

            pct_counts = (
                100.0 * len(overlap) / len(count_id_set)
                if count_id_set else 0.0
            )

            pct_metadata = (
                100.0 * len(overlap) / len(metadata_id_set)
                if metadata_id_set else 0.0
            )

            write_both(report, f"Count-matrix unique cell IDs: {len(count_id_set):,}")
            write_both(report, f"Metadata unique cell IDs:     {len(metadata_id_set):,}")
            write_both(report, f"Matched IDs:                  {len(overlap):,}")
            write_both(report, f"Count IDs matched:            {pct_counts:.2f}%")
            write_both(report, f"Metadata IDs matched:         {pct_metadata:.2f}%")
            write_both(report, f"Only in counts:               {len(only_counts):,}")
            write_both(report, f"Only in metadata:             {len(only_metadata):,}")

        # -------------------------------------------------------------
        # 5. Marker genes
        # -------------------------------------------------------------

        write_both(report, "\n5. MARKER-GENE CHECK")
        write_both(report, "-" * 78)

        gene_lookup = {str(g).upper(): str(g) for g in gene_columns}

        found_markers = []
        missing_markers = []

        for marker in MARKER_GENES:
            key = marker.upper()

            if key in gene_lookup:
                found_markers.append(marker)
                write_both(
                    report,
                    f"[FOUND]   {marker} -> {gene_lookup[key]}"
                )
            else:
                missing_markers.append(marker)
                write_both(report, f"[MISSING] {marker}")

        # Also search for NKX6-related names because naming may differ.
        nkx_matches = [
            g for g in gene_columns
            if "NKX6" in str(g).upper()
        ]

        if nkx_matches:
            write_both(
                report,
                "\nGene columns containing 'NKX6': "
                + ", ".join(map(str, nkx_matches[:20]))
            )

        # -------------------------------------------------------------
        # 6. Endocrine pseudotime metadata
        # -------------------------------------------------------------

        write_both(report, "\n6. ENDOCRINE PSEUDOTIME METADATA")
        write_both(report, "-" * 78)

        pseudo = pd.read_csv(
            PSEUDOTIME_FILE,
            sep="\t",
            compression="gzip",
            low_memory=False,
        )

        write_both(
            report,
            f"Dimensions: {pseudo.shape[0]:,} rows x "
            f"{pseudo.shape[1]:,} columns"
        )

        write_both(report, "\nColumns:")

        for i, col in enumerate(pseudo.columns, start=1):
            write_both(report, f"  {i:2d}. {col}")

        if "CellDay" in pseudo.columns:
            write_both(report, "\nCellDay distribution:")

            for value, count in pseudo["CellDay"].value_counts(
                dropna=False
            ).sort_index().items():
                write_both(
                    report,
                    f"  {str(value):20s} {count:8,d}"
                )

        if "Pseudotime_branch" in pseudo.columns:
            write_both(report, "\nPseudotime_branch distribution:")

            for value, count in pseudo["Pseudotime_branch"].value_counts(
                dropna=False
            ).items():
                write_both(
                    report,
                    f"  {str(value):30s} {count:8,d}"
                )

        # Check overlap between endocrine pseudotime cells and main matrix.
        if "library.barcode" in pseudo.columns:
            pseudo_ids = set(normalize_ids(pseudo["library.barcode"]))
            pseudo_in_counts = pseudo_ids & count_id_set

            pct_pseudo = (
                100.0 * len(pseudo_in_counts) / len(pseudo_ids)
                if pseudo_ids else 0.0
            )

            write_both(
                report,
                f"\nPseudotime unique cell IDs: {len(pseudo_ids):,}"
            )
            write_both(
                report,
                f"Pseudotime IDs found in count matrix: "
                f"{len(pseudo_in_counts):,} ({pct_pseudo:.2f}%)"
            )

        # -------------------------------------------------------------
        # 7. Preflight assessment
        # -------------------------------------------------------------

        write_both(report, "\n7. CORRECTED PREFLIGHT ASSESSMENT")
        write_both(report, "-" * 78)

        overlap_count = (
            len(count_id_set & metadata_id_set)
            if metadata_id_set else 0
        )

        overlap_fraction = (
            overlap_count / len(count_id_set)
            if count_id_set else 0.0
        )

        checks = [
            (
                "All required files available",
                all(
                    path.exists() and path.stat().st_size > 0
                    for _, path in required_files
                ),
            ),
            (
                "Count matrix and metadata have equal row counts",
                n_cells == metadata.shape[0],
            ),
            (
                "At least 10,000 genes available",
                n_genes >= 10000,
            ),
            (
                "At least 90% count/metadata cell-ID overlap",
                overlap_fraction >= 0.90,
            ),
            (
                "Temporal field detected",
                len(time_cols) > 0,
            ),
            (
                "Differentiation/replicate field detected",
                "Differentiation" in metadata.columns,
            ),
            (
                "Cell-type/cluster annotation detected",
                len(celltype_cols) > 0,
            ),
            (
                "At least five key endocrine markers found",
                len(found_markers) >= 5,
            ),
            (
                "Endocrine pseudotime metadata available",
                pseudo.shape[0] > 0,
            ),
        ]

        passed = 0

        for label, status in checks:
            if status:
                write_both(report, f"[PASS] {label}")
                passed += 1
            else:
                write_both(report, f"[CHECK] {label}")

        write_both(
            report,
            f"\nAutomated checks passed: {passed}/{len(checks)}"
        )

        if passed == len(checks):
            write_both(report, "\nPRELIMINARY STATUS: PASS")
            write_both(
                report,
                "GSE114412 Stage 5 passes the structural preflight."
            )
        else:
            write_both(report, "\nPRELIMINARY STATUS: REVIEW REQUIRED")
            write_both(
                report,
                "Review any remaining CHECK items before final approval."
            )

        write_both(
            report,
            "\nNo normalization, filtering, PCA, trajectory inference, "
            "or GDIS calculation was performed."
        )

        write_both(
            report,
            f"\nReport saved to: {REPORT_FILE}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

