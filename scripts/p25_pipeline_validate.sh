#!/usr/bin/env bash
set -euo pipefail

# Validate the established GDIS-Bio P25 pipeline.
#
# Modes:
#   --smoke   : repository/dependency/compile checks; additionally runs one
#               real P25-I profile only when required upstream outputs exist.
#   --verify  : repository/dependency/compile checks; additionally verifies
#               the full P25-I output when it is materialized in this clone.
#
# Clean GitHub clones are not expected to contain large/generated results.
# Missing generated scientific outputs are therefore reported as INFO, not as
# repository-validation failures.

MODE="${1:---smoke}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "$MODE" != "--smoke" && "$MODE" != "--verify" ]]; then
    echo "Usage: bash scripts/p25_pipeline_validate.sh [--smoke|--verify]"
    exit 2
fi

P25_PYCACHE_ROOT="${TMPDIR:-/tmp}/gdis_bio_p25_pycache_${USER:-user}_$$"
export PYTHONPYCACHEPREFIX="$P25_PYCACHE_ROOT"
trap 'rm -rf "$P25_PYCACHE_ROOT"' EXIT

P25_SCRIPTS=(
    "scripts/p25_GSE175634_pseudotime_ordering_robustness.py"
    "scripts/p25b_GSE175634_cell_window_reconstruction_audit.py"
    "scripts/p25c_GSE175634_ordering_perturbation_pilot.py"
    "scripts/p25d_GSE175634_ordering_robustness_full.py"
    "scripts/p25e_GSE175634_ordering_statistical_analysis.py"
    "scripts/p25f_GSE175634_alternative_pseudotime.py"
    "scripts/p25g_GSE175634_graph_geodesic_pilot.py"
    "scripts/p25h_GSE175634_alternative_ordering_gdis.py"
    "scripts/p25i_GSE175634_geodesic_cohort_audit.py"
    "scripts/p25j_GSE175634_geodesic_gdis_comparison.py"
    "scripts/p25k_GSE175634_geodesic_paired_statistics.py"
    "scripts/p25l_make_supplementary_figure_s7_and_tables.py"
    "scripts/p25m_make_figure_s8_and_auxiliary_tables.py"
)

TEMP_DUPLICATES=(
    "scripts/p25j_GSE175634_geodesic_cohort_statistical_summary.py"
    "scripts/p25k_GSE175634_geodesic_cohort_robustness_figure.py"
)

OUT_ROOT="results/p25_GSE175634_pseudotime_ordering_robustness"
ALT_OUT="${OUT_ROOT}/alternative_reconstruction_cohort"
P25_STATE_FILE="results/p15_external_GSE175634_preprocessing_state_space/data/external_state_space_pca_50.csv.gz"
P25_MANIFEST_FILE="results/p17b_external_GSE175634_transition_evaluability_freeze/tables/04_final_primary_gdis_file_manifest.csv"
FULL_I="${ALT_OUT}/01_geodesic_qualification_full.csv"

show_header() {
    echo "============================================================"
    echo "GDIS-Bio P25 PIPELINE VALIDATION"
    echo "Mode:       $MODE"
    echo "Python:     $($PYTHON_BIN --version 2>&1)"
    echo "Repository: $(pwd)"
    echo "============================================================"
}

show_header

if [[ ! -d scripts ]]; then
    echo "ERROR: scripts/ directory not found."
    echo "Run this validator from the GDIS-Bio repository root."
    exit 1
fi

echo
echo "[1/6] Checking required P25 scripts..."
for f in "${P25_SCRIPTS[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing required P25 stage: $f"
        exit 1
    fi
    echo "  OK: $f"
done

for f in "${TEMP_DUPLICATES[@]}"; do
    if [[ -e "$f" ]]; then
        echo "ERROR: temporary duplicate-number script still present: $f"
        exit 1
    fi
done

echo
echo "[2/6] Checking Python dependencies..."
"$PYTHON_BIN" - <<'PY'
mods = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "anndata", "scanpy"]
for name in mods:
    __import__(name)
    print(f"  OK: {name}")
PY

echo
echo "[3/6] Compiling the established P25 pipeline..."
"$PYTHON_BIN" -m py_compile "${P25_SCRIPTS[@]}"
echo "  OK: all established P25 scripts compile"

echo
echo "[4/6] Checking for obvious repository junk..."
junk_found=0
while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    echo "  JUNK: $f"
    junk_found=1
done < <(
    find scripts -maxdepth 1 -type f \( \
        -name "*.before_*" -o \
        -name "*.bak" -o \
        -name "*.backup" -o \
        -name "*.orig" -o \
        -name "*~" -o \
        -name "*.swp" -o \
        -name "*.swo" \
    \) -print
)

if [[ "$junk_found" -ne 0 ]]; then
    echo "ERROR: temporary/backup files remain in scripts/."
    exit 1
fi

echo "  OK: no obvious temporary script artifacts"

echo
echo "[5/6] Pipeline execution/integrity check..."

if [[ "$MODE" == "--smoke" ]]; then
    if [[ -s "$P25_STATE_FILE" && -s "$P25_MANIFEST_FILE" ]]; then
        echo "  Upstream P25-I prerequisites detected; running one real-data profile."

        "$PYTHON_BIN" -u scripts/p25i_GSE175634_geodesic_cohort_audit.py \
            --limit 1 2>&1 | tee p25_pipeline_smoke_i.log

        PILOT="${ALT_OUT}/01_geodesic_qualification_pilot.csv"

        if [[ ! -s "$PILOT" ]]; then
            echo "ERROR: P25-I smoke-test output missing or empty: $PILOT"
            exit 1
        fi

        "$PYTHON_BIN" - "$PILOT" <<'PY'
import sys
import pandas as pd

f = sys.argv[1]
df = pd.read_csv(f, dtype={"individual": str})
required = {"scope", "individual", "qualified", "failure_reason"}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"Missing smoke-test columns: {sorted(missing)}")
if len(df) != 1:
    raise SystemExit(f"Expected 1 smoke-test profile; found {len(df)}")
reason = str(df.iloc[0]["failure_reason"])
if reason.startswith("computation_error"):
    raise SystemExit(f"P25-I smoke test failed computationally: {reason}")
print("  OK: P25-I real-data smoke test completed")
print(df[["scope", "individual", "qualified", "failure_reason"]].to_string(index=False))
PY
        rm -f "$PILOT"
    else
        echo "  INFO: real-data P25-I execution skipped in this clean clone."
        echo "  Required generated upstream inputs are not present:"
        [[ -s "$P25_STATE_FILE" ]] || echo "    MISSING: $P25_STATE_FILE"
        [[ -s "$P25_MANIFEST_FILE" ]] || echo "    MISSING: $P25_MANIFEST_FILE"
        echo "  This is expected for a source-only GitHub checkout."
        echo "  Repository, dependency, and compilation checks passed."
    fi
else
    if [[ ! -s "$FULL_I" ]]; then
        echo "  INFO: full P25-I numerical output is not materialized in this clone:"
        echo "    $FULL_I"
        echo "  This is expected for a source-only GitHub checkout."
        echo "  Repository, dependency, and compilation checks passed."
    else
        "$PYTHON_BIN" - "$FULL_I" <<'PY'
import sys
import pandas as pd

f = sys.argv[1]
df = pd.read_csv(f, dtype={"individual": str})
if len(df) != 48:
    raise SystemExit(f"Expected 48 P25-I profiles; found {len(df)}")
errors = df["failure_reason"].fillna("").astype(str).str.startswith("computation_error")
if errors.any():
    bad = df.loc[errors, ["scope", "individual", "failure_reason"]]
    raise SystemExit("P25-I contains computation errors:\n" + bad.to_string(index=False))
expected_scopes = {"cf_extension", "cm_extension", "shared_backbone"}
actual_scopes = set(df["scope"].astype(str))
if actual_scopes != expected_scopes:
    raise SystemExit(f"Unexpected P25-I scopes: {sorted(actual_scopes)}")
print("  OK: P25-I full cohort contains 48 profiles")
print("  OK: no P25-I computation_error records")
print("  OK: expected scopes are present")
PY

        n_csv=$(find "$OUT_ROOT" -type f -name "*.csv" -size +0c | wc -l)
        n_png=$(find "$OUT_ROOT" -type f -name "*.png" -size +0c | wc -l)
        n_pdf=$(find "$OUT_ROOT" -type f -name "*.pdf" -size +0c | wc -l)
        echo "  Non-empty CSV outputs: $n_csv"
        echo "  Non-empty PNG outputs: $n_png"
        echo "  Non-empty PDF outputs: $n_pdf"
    fi
fi

echo
echo "[6/6] Final validation..."
"$PYTHON_BIN" -m py_compile "${P25_SCRIPTS[@]}"

echo
echo "============================================================"
echo "P25 PIPELINE VALIDATION PASSED"
echo "============================================================"
echo
echo "Established P25 stages retained:"
printf '  %s\n' "${P25_SCRIPTS[@]}"
