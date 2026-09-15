#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================
"""
p16_GSE175634_state_space_geometry_validation.py

Pre-GDIS geometry validation of the frozen GSE175634 external-validation
state space.

Scientific role
---------------
The discovery analysis used 50 PCs as the primary state space. External
validation therefore TRANSFERS 50 PCs as the primary representation rather
than selecting a dimension after seeing external GDIS behavior.

This script asks whether the transferred 50-PC state space is:
    - numerically well behaved,
    - not pathologically distance-concentrated,
    - locally structured,
    - converged relative to lower-dimensional prefixes,
    - biologically coherent across independent individuals.

Lower-dimensional representations (5/10/20/30 PCs) remain frozen sensitivity
analyses. They are NOT candidates for tuning GDIS.

No GDIS values are calculated.

Input
-----
results/p15_external_GSE175634_preprocessing_state_space/data/
    external_state_space_pca_50.csv.gz

The 5/10/20/30-PC representations are exact prefixes of the common 50-PC PCA
constructed in p15, so this script loads the 50-PC file once and evaluates
the corresponding prefixes directly.

Diagnostics
-----------
A. Random-pair distance geometry across dimensions.
B. 50-PC distance concentration.
C. Stratified-sample exact 30-nearest-neighbor geometry.
D. Neighbor preservation from lower dimensions to 50 PCs.
E. Individual x biological-state centroid geometry.
F. Leave-one-individual-out classification of shared-backbone state centroids.
G. Directional reproducibility of biological state-transition vectors.

The stratified kNN analysis is a geometry diagnostic only. It does not alter
the frozen cell cohorts.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import balanced_accuracy_score
except ImportError as exc:
    raise SystemExit(
        "ERROR: scikit-learn is required.\n"
        "Install with: python -m pip install scikit-learn"
    ) from exc


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR

P15_DIR = (
    PROJECT_DIR
    / "results"
    / "p15_external_GSE175634_preprocessing_state_space"
)

P15_DATA_DIR = P15_DIR / "data"

STATE_SPACE_FILE = (
    P15_DATA_DIR
    / "external_state_space_pca_50.csv.gz"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "p16_external_GSE175634_state_space_geometry_validation"
)

TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"

REPORT_FILE = (
    RESULTS_DIR
    / "p16_external_GSE175634_state_space_geometry_validation_report.txt"
)


# ---------------------------------------------------------------------
# Frozen analysis settings
# ---------------------------------------------------------------------

DIMENSIONS = [5, 10, 20, 30, 50]
PRIMARY_DIMENSION = 50

RANDOM_SEED = 20260913

# Random-pair distance diagnostic.
N_RANDOM_PAIRS = 250_000
PAIR_CHUNK_SIZE = 25_000

# Exact local-neighborhood diagnostic on a deterministic stratified subset.
K_NEIGHBORS = 30
STRATIFIED_SAMPLE_PER_INDIVIDUAL_STATE = 35

CORE_STATES = [
    "IPSC",
    "MES",
    "CMES",
    "PROG",
]

ALL_STATES = [
    "IPSC",
    "MES",
    "CMES",
    "PROG",
    "CM",
    "CF",
]

SEGMENTS = [
    ("T1", "IPSC", "MES"),
    ("T2", "MES", "CMES"),
    ("CORE3", "CMES", "PROG"),
    ("T3_CM", "PROG", "CM"),
    ("T3_CF", "PROG", "CF"),
]

# Pre-specified geometry qualification thresholds.
REFERENCE_DISTANCE_CV_MIN = 0.10
REFERENCE_Q95_Q05_MIN = 1.50
REFERENCE_NN_RATIO_MIN = 1.10
REFERENCE_LOCAL_SCALE_CV_MIN = 0.05
DISTANCE_RHO_30_TO_50_MIN = 0.98
NEIGHBOR_OVERLAP_30_TO_50_MIN = 0.50
CENTROID_DISTANCE_RHO_30_TO_50_MIN = 0.98
CORE_LOIO_BALANCED_ACCURACY_50_MIN = 0.75
STATE_SEPARATION_RATIO_50_MIN = 1.00


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def write_both(report, text=""):
    print(text)
    report.write(text + "\n")
    report.flush()


def save_table(df, filename, index=True):
    path = TABLE_DIR / filename
    df.to_csv(path, index=index)
    return path


def safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 3:
        return np.nan

    return float(
        spearmanr(
            x[mask],
            y[mask],
        ).statistic
    )


def coefficient_of_variation(values):
    values = np.asarray(values, dtype=float)

    mean = float(
        np.mean(values)
    )

    if mean == 0:
        return np.nan

    return float(
        np.std(
            values,
            ddof=1,
        )
        / mean
    )


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return np.nan

    return float(
        np.dot(a, b)
        / denominator
    )


def create_random_pairs(n_cells, n_pairs, rng):
    """
    Create deterministic random cell-index pairs with i != j.
    """
    i = rng.integers(
        0,
        n_cells,
        size=n_pairs,
        dtype=np.int64,
    )

    j = rng.integers(
        0,
        n_cells,
        size=n_pairs,
        dtype=np.int64,
    )

    same = (
        i == j
    )

    while np.any(same):
        j[same] = rng.integers(
            0,
            n_cells,
            size=int(
                same.sum()
            ),
            dtype=np.int64,
        )

        same = (
            i == j
        )

    return i, j


def pair_distances(
    pcs,
    i,
    j,
    n_dim,
):
    """
    Euclidean distances for fixed random pairs, computed in chunks.
    """
    output = np.empty(
        len(i),
        dtype=np.float64,
    )

    for start in range(
        0,
        len(i),
        PAIR_CHUNK_SIZE,
    ):

        stop = min(
            start
            + PAIR_CHUNK_SIZE,
            len(i),
        )

        diff = (
            pcs[
                i[start:stop],
                :n_dim,
            ].astype(
                np.float64,
                copy=False,
            )
            - pcs[
                j[start:stop],
                :n_dim,
            ].astype(
                np.float64,
                copy=False,
            )
        )

        output[
            start:stop
        ] = np.sqrt(
            np.sum(
                diff
                * diff,
                axis=1,
            )
        )

    return output


def distance_summary(
    values,
    n_dim,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    q05 = float(
        np.quantile(
            values,
            0.05,
        )
    )

    q95 = float(
        np.quantile(
            values,
            0.95,
        )
    )

    return {
        "dimension":
            n_dim,
        "mean_distance":
            float(
                np.mean(values)
            ),
        "sd_distance":
            float(
                np.std(
                    values,
                    ddof=1,
                )
            ),
        "distance_cv":
            coefficient_of_variation(
                values
            ),
        "q05":
            q05,
        "median":
            float(
                np.median(values)
            ),
        "q95":
            q95,
        "q95_q05_ratio":
            (
                q95 / q05
                if q05 > 0
                else np.nan
            ),
    }


def stratified_sample_indices(
    metadata,
    rng,
):
    """
    Sample up to a fixed number of cells per individual x biological state.
    """
    selected = []

    groups = (
        metadata.groupby(
            [
                "individual",
                "type",
            ],
            observed=True,
            sort=True,
        )
    )

    for _, group in groups:

        positions = group.index.to_numpy(
            dtype=np.int64
        )

        n_take = min(
            STRATIFIED_SAMPLE_PER_INDIVIDUAL_STATE,
            len(
                positions
            ),
        )

        chosen = rng.choice(
            positions,
            size=n_take,
            replace=False,
        )

        selected.extend(
            chosen.tolist()
        )

    return np.asarray(
        sorted(
            selected
        ),
        dtype=np.int64,
    )


def exact_knn(
    X,
    k,
):
    """
    Exact Euclidean kNN on the stratified geometry sample.
    """
    model = NearestNeighbors(
        n_neighbors=k + 1,
        algorithm="brute",
        metric="euclidean",
        n_jobs=-1,
    )

    model.fit(X)

    distances, indices = (
        model.kneighbors(
            X,
            return_distance=True,
        )
    )

    n = X.shape[0]

    clean_indices = np.empty(
        (
            n,
            k,
        ),
        dtype=np.int64,
    )

    clean_distances = np.empty(
        (
            n,
            k,
        ),
        dtype=np.float64,
    )

    for row in range(n):

        mask = (
            indices[row]
            != row
        )

        row_indices = indices[
            row
        ][mask]

        row_distances = distances[
            row
        ][mask]

        if len(
            row_indices
        ) < k:
            raise RuntimeError(
                "Could not obtain enough non-self "
                "nearest neighbors."
            )

        clean_indices[
            row,
            :,
        ] = row_indices[
            :k
        ]

        clean_distances[
            row,
            :,
        ] = row_distances[
            :k
        ]

    return (
        clean_distances,
        clean_indices,
    )


def mean_neighbor_overlap(
    neighbors_a,
    neighbors_b,
):
    """
    Mean fraction of shared neighbors per cell.
    """
    if (
        neighbors_a.shape
        != neighbors_b.shape
    ):
        raise ValueError(
            "Neighbor arrays must have identical shapes."
        )

    k = neighbors_a.shape[1]

    overlaps = np.empty(
        neighbors_a.shape[0],
        dtype=float,
    )

    for row in range(
        neighbors_a.shape[0]
    ):

        overlaps[row] = (
            len(
                set(
                    neighbors_a[
                        row
                    ].tolist()
                ).intersection(
                    neighbors_b[
                        row
                    ].tolist()
                )
            )
            / k
        )

    return float(
        np.mean(
            overlaps
        )
    )


def build_centroid_table(
    df,
    pc_columns,
):
    """
    Median PC centroid for each individual x state.
    """
    columns = [
        "individual",
        "type",
    ] + pc_columns

    centroid = (
        df[
            columns
        ]
        .groupby(
            [
                "individual",
                "type",
            ],
            observed=True,
        )[
            pc_columns
        ]
        .median()
        .reset_index()
    )

    return centroid


def centroid_distance_vector(
    centroid,
    n_dim,
):
    pc_columns = [
        f"PC{i}"
        for i in range(
            1,
            n_dim + 1,
        )
    ]

    matrix = centroid[
        pc_columns
    ].to_numpy(
        dtype=float
    )

    return pdist(
        matrix,
        metric="euclidean",
    )


def global_state_centroids(
    df,
    n_dim,
):
    pc_columns = [
        f"PC{i}"
        for i in range(
            1,
            n_dim + 1,
        )
    ]

    result = (
        df[
            [
                "type",
            ]
            + pc_columns
        ]
        .groupby(
            "type",
            observed=True,
        )[
            pc_columns
        ]
        .median()
    )

    return result


def state_separation_ratio(
    individual_centroids,
    global_centroids,
    n_dim,
):
    """
    Ratio:
        median distance between global biological-state centroids
        ---------------------------------------------------------
        median distance between individual centroids of the same state

    Values > 1 indicate biological-state separation exceeds typical
    between-individual centroid dispersion.
    """
    pc_columns = [
        f"PC{i}"
        for i in range(
            1,
            n_dim + 1,
        )
    ]

    global_matrix = (
        global_centroids[
            pc_columns
        ].to_numpy(
            dtype=float
        )
    )

    between_state = pdist(
        global_matrix,
        metric="euclidean",
    )

    within_state_distances = []

    for state, group in (
        individual_centroids.groupby(
            "type",
            observed=True,
        )
    ):

        if len(
            group
        ) < 2:
            continue

        matrix = group[
            pc_columns
        ].to_numpy(
            dtype=float
        )

        values = pdist(
            matrix,
            metric="euclidean",
        )

        within_state_distances.extend(
            values.tolist()
        )

    if not within_state_distances:
        return np.nan

    within_median = float(
        np.median(
            np.asarray(
                within_state_distances,
                dtype=float,
            )
        )
    )

    between_median = float(
        np.median(
            between_state
        )
    )

    return (
        between_median
        / within_median
        if within_median > 0
        else np.nan
    )


def leave_one_individual_out_core_classification(
    individual_centroids,
    n_dim,
):
    """
    Classify each held-out individual's four shared-backbone state centroids
    by nearest training-state centroid.

    Each individual contributes one centroid per frozen core state, so this
    evaluates whether state geometry generalizes across individuals without
    using cell-count weighting.
    """
    pc_columns = [
        f"PC{i}"
        for i in range(
            1,
            n_dim + 1,
        )
    ]

    core = individual_centroids[
        individual_centroids[
            "type"
        ].isin(
            CORE_STATES
        )
    ].copy()

    individuals = sorted(
        core[
            "individual"
        ].unique()
        .tolist()
    )

    y_true = []
    y_pred = []

    per_individual_rows = []

    for individual in individuals:

        train = core[
            core[
                "individual"
            ] != individual
        ]

        test = core[
            core[
                "individual"
            ] == individual
        ]

        # p14 froze all 19 individuals as core eligible.
        present = set(
            test[
                "type"
            ].tolist()
        )

        if not set(
            CORE_STATES
        ).issubset(
            present
        ):
            continue

        training_centroids = (
            train.groupby(
                "type",
                observed=True,
            )[
                pc_columns
            ]
            .mean()
            .reindex(
                CORE_STATES
            )
        )

        correct = 0
        total = 0

        for _, row in (
            test.iterrows()
        ):

            true_state = str(
                row[
                    "type"
                ]
            )

            if (
                true_state
                not in CORE_STATES
            ):
                continue

            point = row[
                pc_columns
            ].to_numpy(
                dtype=float
            )

            matrix = (
                training_centroids[
                    pc_columns
                ].to_numpy(
                    dtype=float
                )
            )

            distances = np.linalg.norm(
                matrix
                - point,
                axis=1,
            )

            predicted = CORE_STATES[
                int(
                    np.argmin(
                        distances
                    )
                )
            ]

            y_true.append(
                true_state
            )

            y_pred.append(
                predicted
            )

            total += 1

            if (
                predicted
                == true_state
            ):
                correct += 1

        per_individual_rows.append(
            {
                "individual":
                    individual,
                "dimension":
                    n_dim,
                "n_test_state_centroids":
                    total,
                "accuracy":
                    (
                        correct / total
                        if total
                        else np.nan
                    ),
            }
        )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
        if y_true
        else np.nan
    )

    overall_accuracy = (
        float(
            np.mean(
                np.asarray(
                    y_true
                )
                == np.asarray(
                    y_pred
                )
            )
        )
        if y_true
        else np.nan
    )

    return (
        {
            "dimension":
                n_dim,
            "n_classified_centroids":
                len(
                    y_true
                ),
            "overall_accuracy":
                overall_accuracy,
            "balanced_accuracy":
                float(
                    balanced_accuracy
                ),
        },
        pd.DataFrame(
            per_individual_rows
        ),
    )


def segment_direction_consistency(
    individual_centroids,
    n_dim,
):
    """
    Compare each individual's biological-transition vector with the
    corresponding global transition vector.
    """
    pc_columns = [
        f"PC{i}"
        for i in range(
            1,
            n_dim + 1,
        )
    ]

    global_by_state = (
        individual_centroids.groupby(
            "type",
            observed=True,
        )[
            pc_columns
        ]
        .mean()
    )

    rows = []

    for (
        segment_id,
        source,
        destination,
    ) in SEGMENTS:

        if (
            source
            not in global_by_state.index
            or destination
            not in global_by_state.index
        ):
            continue

        global_vector = (
            global_by_state.loc[
                destination
            ].to_numpy(
                dtype=float
            )
            - global_by_state.loc[
                source
            ].to_numpy(
                dtype=float
            )
        )

        cosine_values = []

        for individual, group in (
            individual_centroids.groupby(
                "individual",
                observed=True,
            )
        ):

            indexed = group.set_index(
                "type"
            )

            if (
                source
                not in indexed.index
                or destination
                not in indexed.index
            ):
                continue

            vector = (
                indexed.loc[
                    destination,
                    pc_columns,
                ].to_numpy(
                    dtype=float
                )
                - indexed.loc[
                    source,
                    pc_columns,
                ].to_numpy(
                    dtype=float
                )
            )

            cosine = cosine_similarity(
                vector,
                global_vector,
            )

            if np.isfinite(
                cosine
            ):
                cosine_values.append(
                    cosine
                )

        values = np.asarray(
            cosine_values,
            dtype=float,
        )

        rows.append(
            {
                "dimension":
                    n_dim,
                "segment_id":
                    segment_id,
                "source_state":
                    source,
                "destination_state":
                    destination,
                "n_individuals":
                    len(
                        values
                    ),
                "median_cosine":
                    (
                        float(
                            np.median(
                                values
                            )
                        )
                        if len(
                            values
                        )
                        else np.nan
                    ),
                "q25_cosine":
                    (
                        float(
                            np.quantile(
                                values,
                                0.25,
                            )
                        )
                        if len(
                            values
                        )
                        else np.nan
                    ),
                "q75_cosine":
                    (
                        float(
                            np.quantile(
                                values,
                                0.75,
                            )
                        )
                        if len(
                            values
                        )
                        else np.nan
                    ),
                "fraction_positive_cosine":
                    (
                        float(
                            np.mean(
                                values
                                > 0
                            )
                        )
                        if len(
                            values
                        )
                        else np.nan
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def plot_metric_by_dimension(
    table,
    x_column,
    y_column,
    ylabel,
    title,
    output_path,
):
    fig, ax = plt.subplots(
        figsize=(8.2, 5.4)
    )

    ax.plot(
        table[
            x_column
        ],
        table[
            y_column
        ],
        marker="o",
        linewidth=1.6,
    )

    ax.set_xlabel(
        "Number of PCs"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    for directory in [
        TABLE_DIR,
        FIGURE_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    if not STATE_SPACE_FILE.exists():
        raise FileNotFoundError(
            f"Missing p15 50-PC state space: "
            f"{STATE_SPACE_FILE}"
        )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as report:

        write_both(
            report,
            "=" * 98,
        )

        write_both(
            report,
            "GDIS-Bio External Validation State-Space Geometry Validation",
        )

        write_both(
            report,
            "Dataset: GSE175634",
        )

        write_both(
            report,
            "=" * 98,
        )

        write_both(
            report,
            "\n50 PCs are transferred from the discovery workflow "
            "as the primary external-validation representation.",
        )

        write_both(
            report,
            "Lower-dimensional prefixes are sensitivity analyses only.",
        )

        write_both(
            report,
            "NO GDIS VALUES ARE CALCULATED.",
        )

        # =========================================================
        # 1. Load and validate state space
        # =========================================================

        write_both(
            report,
            "\n1. INPUT VALIDATION",
        )

        write_both(
            report,
            "-" * 98,
        )

        df = pd.read_csv(
            STATE_SPACE_FILE,
            compression="gzip",
            low_memory=False,
        )

        required_metadata = [
            "cell",
            "individual",
            "type",
            "diffday",
            "dpt_pseudotime",
        ]

        pc_columns_50 = [
            f"PC{i}"
            for i in range(
                1,
                51,
            )
        ]

        missing = [
            column
            for column in (
                required_metadata
                + pc_columns_50
            )
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                "Missing required columns: "
                + ", ".join(
                    missing
                )
            )

        df[
            "cell"
        ] = df[
            "cell"
        ].astype(str)

        df[
            "individual"
        ] = df[
            "individual"
        ].astype(str)

        df[
            "type"
        ] = df[
            "type"
        ].astype(str)

        df[
            "dpt_pseudotime"
        ] = pd.to_numeric(
            df[
                "dpt_pseudotime"
            ],
            errors="raise",
        )

        for column in pc_columns_50:
            df[
                column
            ] = pd.to_numeric(
                df[
                    column
                ],
                errors="raise",
            ).astype(
                np.float32
            )

        pcs = df[
            pc_columns_50
        ].to_numpy(
            dtype=np.float32,
            copy=True,
        )

        finite = bool(
            np.isfinite(
                pcs
            ).all()
        )

        write_both(
            report,
            f"Cells: "
            f"{len(df):,}",
        )

        write_both(
            report,
            f"Individuals: "
            f"{df['individual'].nunique()}",
        )

        write_both(
            report,
            f"Cell states: "
            f"{sorted(df['type'].unique().tolist())}",
        )

        write_both(
            report,
            f"PC dimensions: "
            f"{pcs.shape[1]}",
        )

        write_both(
            report,
            f"Finite PC coordinates: "
            f"{finite}",
        )

        # =========================================================
        # 2. Random-pair distance geometry
        # =========================================================

        write_both(
            report,
            "\n2. RANDOM-PAIR DISTANCE GEOMETRY",
        )

        write_both(
            report,
            "-" * 98,
        )

        rng = np.random.default_rng(
            RANDOM_SEED
        )

        pair_i, pair_j = (
            create_random_pairs(
                len(df),
                N_RANDOM_PAIRS,
                rng,
            )
        )

        distance_vectors = {}
        distance_summary_rows = []

        for n_dim in DIMENSIONS:

            distances = pair_distances(
                pcs,
                pair_i,
                pair_j,
                n_dim,
            )

            distance_vectors[
                n_dim
            ] = distances

            summary = distance_summary(
                distances,
                n_dim,
            )

            distance_summary_rows.append(
                summary
            )

            write_both(
                report,
                f"{n_dim:>2} PCs: "
                f"mean={summary['mean_distance']:.6f}, "
                f"CV={summary['distance_cv']:.6f}, "
                f"q95/q05={summary['q95_q05_ratio']:.6f}",
            )

        distance_summary_table = pd.DataFrame(
            distance_summary_rows
        )

        save_table(
            distance_summary_table,
            "01_random_pair_distance_summary.csv",
            index=False,
        )

        convergence_rows = []

        for n_dim in DIMENSIONS:

            rho_to_50 = safe_spearman(
                distance_vectors[
                    n_dim
                ],
                distance_vectors[
                    50
                ],
            )

            convergence_rows.append(
                {
                    "dimension":
                        n_dim,
                    "spearman_distance_vs_50pc":
                        rho_to_50,
                }
            )

        adjacent_pairs = [
            (5, 10),
            (10, 20),
            (20, 30),
            (30, 50),
        ]

        adjacent_rows = []

        for lower, upper in adjacent_pairs:

            adjacent_rows.append(
                {
                    "lower_dimension":
                        lower,
                    "upper_dimension":
                        upper,
                    "spearman_distance_correlation":
                        safe_spearman(
                            distance_vectors[
                                lower
                            ],
                            distance_vectors[
                                upper
                            ],
                        ),
                }
            )

        distance_convergence = pd.DataFrame(
            convergence_rows
        )

        adjacent_distance_convergence = pd.DataFrame(
            adjacent_rows
        )

        save_table(
            distance_convergence,
            "02_distance_convergence_to_50pc.csv",
            index=False,
        )

        save_table(
            adjacent_distance_convergence,
            "03_adjacent_dimension_distance_convergence.csv",
            index=False,
        )

        write_both(
            report,
            "\nDistance convergence to 50 PCs:\n"
            + distance_convergence.round(
                6
            ).to_string(
                index=False
            ),
        )

        write_both(
            report,
            "\nAdjacent-dimensional convergence:\n"
            + adjacent_distance_convergence.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 3. Exact local-neighborhood geometry on stratified sample
        # =========================================================

        write_both(
            report,
            "\n3. LOCAL NEIGHBORHOOD GEOMETRY "
            "(STRATIFIED EXACT-kNN SAMPLE)",
        )

        write_both(
            report,
            "-" * 98,
        )

        sample_indices = stratified_sample_indices(
            df[
                [
                    "individual",
                    "type",
                ]
            ],
            rng,
        )

        sample_df = df.iloc[
            sample_indices
        ].reset_index(
            drop=True
        )

        sample_pcs = pcs[
            sample_indices,
            :,
        ]

        write_both(
            report,
            f"Stratified geometry sample: "
            f"{len(sample_df):,} cells",
        )

        write_both(
            report,
            f"Target per individual x state: "
            f"{STRATIFIED_SAMPLE_PER_INDIVIDUAL_STATE}",
        )

        knn_neighbors = {}
        knn_distances = {}
        knn_rows = []

        for n_dim in DIMENSIONS:

            distances, neighbors = exact_knn(
                sample_pcs[
                    :,
                    :n_dim,
                ],
                K_NEIGHBORS,
            )

            knn_neighbors[
                n_dim
            ] = neighbors

            knn_distances[
                n_dim
            ] = distances

            first_nn = distances[
                :,
                0,
            ]

            kth_nn = distances[
                :,
                -1,
            ]

            local_scale = np.mean(
                distances,
                axis=1,
            )

            median_first = float(
                np.median(
                    first_nn
                )
            )

            median_kth = float(
                np.median(
                    kth_nn
                )
            )

            ratio = (
                median_kth
                / median_first
                if median_first > 0
                else np.nan
            )

            local_cv = coefficient_of_variation(
                local_scale
            )

            pseudo_rho = safe_spearman(
                local_scale,
                sample_df[
                    "dpt_pseudotime"
                ].to_numpy(
                    dtype=float
                ),
            )

            knn_rows.append(
                {
                    "dimension":
                        n_dim,
                    "sample_cells":
                        len(
                            sample_df
                        ),
                    "k":
                        K_NEIGHBORS,
                    "median_first_neighbor_distance":
                        median_first,
                    "median_kth_neighbor_distance":
                        median_kth,
                    "median_kth_to_first_ratio":
                        ratio,
                    "mean_local_scale":
                        float(
                            np.mean(
                                local_scale
                            )
                        ),
                    "local_scale_cv":
                        local_cv,
                    "spearman_local_scale_vs_pseudotime":
                        pseudo_rho,
                }
            )

        knn_table = pd.DataFrame(
            knn_rows
        )

        save_table(
            knn_table,
            "04_local_knn_geometry_by_dimension.csv",
            index=False,
        )

        write_both(
            report,
            knn_table.round(
                6
            ).to_string(
                index=False
            ),
        )

        overlap_rows = []

        for n_dim in DIMENSIONS:

            overlap_rows.append(
                {
                    "dimension":
                        n_dim,
                    "mean_30nn_overlap_vs_50pc":
                        mean_neighbor_overlap(
                            knn_neighbors[
                                n_dim
                            ],
                            knn_neighbors[
                                50
                            ],
                        ),
                }
            )

        overlap_to_50 = pd.DataFrame(
            overlap_rows
        )

        adjacent_overlap_rows = []

        for lower, upper in adjacent_pairs:

            adjacent_overlap_rows.append(
                {
                    "lower_dimension":
                        lower,
                    "upper_dimension":
                        upper,
                    "mean_30nn_overlap":
                        mean_neighbor_overlap(
                            knn_neighbors[
                                lower
                            ],
                            knn_neighbors[
                                upper
                            ],
                        ),
                }
            )

        adjacent_overlap = pd.DataFrame(
            adjacent_overlap_rows
        )

        save_table(
            overlap_to_50,
            "05_neighbor_overlap_to_50pc.csv",
            index=False,
        )

        save_table(
            adjacent_overlap,
            "06_adjacent_dimension_neighbor_overlap.csv",
            index=False,
        )

        write_both(
            report,
            "\nNeighbor overlap versus 50 PCs:\n"
            + overlap_to_50.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 4. Individual x state centroid geometry
        # =========================================================

        write_both(
            report,
            "\n4. INDIVIDUAL x STATE CENTROID GEOMETRY",
        )

        write_both(
            report,
            "-" * 98,
        )

        individual_centroids_50 = (
            build_centroid_table(
                df,
                pc_columns_50,
            )
        )

        save_table(
            individual_centroids_50,
            "07_individual_state_centroids_50pc.csv",
            index=False,
        )

        centroid_distance_vectors = {}
        centroid_geometry_rows = []

        for n_dim in DIMENSIONS:

            vector = centroid_distance_vector(
                individual_centroids_50,
                n_dim,
            )

            centroid_distance_vectors[
                n_dim
            ] = vector

            global_centroids = (
                global_state_centroids(
                    df,
                    n_dim,
                )
            )

            separation_ratio = (
                state_separation_ratio(
                    individual_centroids_50,
                    global_centroids,
                    n_dim,
                )
            )

            centroid_geometry_rows.append(
                {
                    "dimension":
                        n_dim,
                    "spearman_centroid_distance_vs_50pc":
                        safe_spearman(
                            vector,
                            centroid_distance_vectors[
                                50
                            ]
                            if 50
                            in centroid_distance_vectors
                            else vector,
                        ),
                    "state_separation_to_individual_dispersion_ratio":
                        separation_ratio,
                }
            )

        # Recompute correlations now that the 50-PC vector definitely exists.
        for row in centroid_geometry_rows:

            n_dim = int(
                row[
                    "dimension"
                ]
            )

            row[
                "spearman_centroid_distance_vs_50pc"
            ] = safe_spearman(
                centroid_distance_vectors[
                    n_dim
                ],
                centroid_distance_vectors[
                    50
                ],
            )

        centroid_geometry = pd.DataFrame(
            centroid_geometry_rows
        )

        save_table(
            centroid_geometry,
            "08_centroid_geometry_by_dimension.csv",
            index=False,
        )

        write_both(
            report,
            centroid_geometry.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 5. Leave-one-individual-out biological-state generalization
        # =========================================================

        write_both(
            report,
            "\n5. LEAVE-ONE-INDIVIDUAL-OUT CORE-STATE GENERALIZATION",
        )

        write_both(
            report,
            "-" * 98,
        )

        classification_rows = []
        classification_individual_tables = []

        for n_dim in DIMENSIONS:

            summary, per_individual = (
                leave_one_individual_out_core_classification(
                    individual_centroids_50,
                    n_dim,
                )
            )

            classification_rows.append(
                summary
            )

            classification_individual_tables.append(
                per_individual
            )

        classification_table = pd.DataFrame(
            classification_rows
        )

        classification_by_individual = pd.concat(
            classification_individual_tables,
            ignore_index=True,
        )

        save_table(
            classification_table,
            "09_core_state_loio_classification_by_dimension.csv",
            index=False,
        )

        save_table(
            classification_by_individual,
            "10_core_state_loio_classification_by_individual.csv",
            index=False,
        )

        write_both(
            report,
            classification_table.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 6. Biological-transition vector reproducibility
        # =========================================================

        write_both(
            report,
            "\n6. BIOLOGICAL-TRANSITION VECTOR REPRODUCIBILITY",
        )

        write_both(
            report,
            "-" * 98,
        )

        segment_tables = []

        for n_dim in DIMENSIONS:

            segment_tables.append(
                segment_direction_consistency(
                    individual_centroids_50,
                    n_dim,
                )
            )

        segment_table = pd.concat(
            segment_tables,
            ignore_index=True,
        )

        save_table(
            segment_table,
            "11_transition_vector_direction_consistency.csv",
            index=False,
        )

        write_both(
            report,
            segment_table.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 7. Qualification of transferred 50-PC primary space
        # =========================================================

        write_both(
            report,
            "\n7. TRANSFERRED 50-PC PRIMARY-SPACE QUALIFICATION",
        )

        write_both(
            report,
            "-" * 98,
        )

        distance_50 = (
            distance_summary_table[
                distance_summary_table[
                    "dimension"
                ] == 50
            ].iloc[0]
        )

        knn_50 = (
            knn_table[
                knn_table[
                    "dimension"
                ] == 50
            ].iloc[0]
        )

        rho_30_50 = float(
            adjacent_distance_convergence.loc[
                (
                    adjacent_distance_convergence[
                        "lower_dimension"
                    ] == 30
                )
                & (
                    adjacent_distance_convergence[
                        "upper_dimension"
                    ] == 50
                ),
                "spearman_distance_correlation",
            ].iloc[0]
        )

        overlap_30_50 = float(
            adjacent_overlap.loc[
                (
                    adjacent_overlap[
                        "lower_dimension"
                    ] == 30
                )
                & (
                    adjacent_overlap[
                        "upper_dimension"
                    ] == 50
                ),
                "mean_30nn_overlap",
            ].iloc[0]
        )

        centroid_rho_30_50 = float(
            centroid_geometry.loc[
                centroid_geometry[
                    "dimension"
                ] == 30,
                "spearman_centroid_distance_vs_50pc",
            ].iloc[0]
        )

        state_separation_50 = float(
            centroid_geometry.loc[
                centroid_geometry[
                    "dimension"
                ] == 50,
                "state_separation_to_individual_dispersion_ratio",
            ].iloc[0]
        )

        core_balanced_acc_50 = float(
            classification_table.loc[
                classification_table[
                    "dimension"
                ] == 50,
                "balanced_accuracy",
            ].iloc[0]
        )

        checks = [
            (
                f"50-PC pairwise-distance CV >= "
                f"{REFERENCE_DISTANCE_CV_MIN}",
                float(
                    distance_50[
                        "distance_cv"
                    ]
                )
                >= REFERENCE_DISTANCE_CV_MIN,
            ),
            (
                f"50-PC q95/q05 distance ratio >= "
                f"{REFERENCE_Q95_Q05_MIN}",
                float(
                    distance_50[
                        "q95_q05_ratio"
                    ]
                )
                >= REFERENCE_Q95_Q05_MIN,
            ),
            (
                f"50-PC median {K_NEIGHBORS}th/1st "
                f"neighbor ratio >= "
                f"{REFERENCE_NN_RATIO_MIN}",
                float(
                    knn_50[
                        "median_kth_to_first_ratio"
                    ]
                )
                >= REFERENCE_NN_RATIO_MIN,
            ),
            (
                f"50-PC local-scale CV >= "
                f"{REFERENCE_LOCAL_SCALE_CV_MIN}",
                float(
                    knn_50[
                        "local_scale_cv"
                    ]
                )
                >= REFERENCE_LOCAL_SCALE_CV_MIN,
            ),
            (
                f"30->50 random-pair distance Spearman >= "
                f"{DISTANCE_RHO_30_TO_50_MIN}",
                rho_30_50
                >= DISTANCE_RHO_30_TO_50_MIN,
            ),
            (
                f"30->50 mean 30-NN overlap >= "
                f"{NEIGHBOR_OVERLAP_30_TO_50_MIN}",
                overlap_30_50
                >= NEIGHBOR_OVERLAP_30_TO_50_MIN,
            ),
            (
                f"30-vs-50 individual-state centroid-distance "
                f"Spearman >= "
                f"{CENTROID_DISTANCE_RHO_30_TO_50_MIN}",
                centroid_rho_30_50
                >= CENTROID_DISTANCE_RHO_30_TO_50_MIN,
            ),
            (
                f"50-PC leave-one-individual-out core-state "
                f"balanced accuracy >= "
                f"{CORE_LOIO_BALANCED_ACCURACY_50_MIN}",
                core_balanced_acc_50
                >= CORE_LOIO_BALANCED_ACCURACY_50_MIN,
            ),
            (
                f"50-PC biological-state separation / "
                f"individual-dispersion ratio >= "
                f"{STATE_SEPARATION_RATIO_50_MIN}",
                state_separation_50
                >= STATE_SEPARATION_RATIO_50_MIN,
            ),
        ]

        qualification_rows = []

        passed = 0

        for label, status in checks:

            if status:
                prefix = "[PASS]"
                passed += 1
            else:
                prefix = "[REVIEW]"

            write_both(
                report,
                f"{prefix} {label}",
            )

            qualification_rows.append(
                {
                    "check":
                        label,
                    "pass":
                        bool(
                            status
                        ),
                }
            )

        qualification_table = pd.DataFrame(
            qualification_rows
        )

        save_table(
            qualification_table,
            "12_reference_50pc_qualification.csv",
            index=False,
        )

        write_both(
            report,
            f"\nGeometry checks passed: "
            f"{passed}/{len(checks)}",
        )

        # =========================================================
        # 8. Figures
        # =========================================================

        write_both(
            report,
            "\n8. FIGURES",
        )

        write_both(
            report,
            "-" * 98,
        )

        plot_metric_by_dimension(
            distance_summary_table,
            "dimension",
            "distance_cv",
            "Pairwise-distance coefficient of variation",
            "Distance concentration across PCA dimensions",
            FIGURE_DIR
            / "01_distance_cv_by_dimension.png",
        )

        plot_metric_by_dimension(
            overlap_to_50,
            "dimension",
            "mean_30nn_overlap_vs_50pc",
            "Mean 30-NN overlap with 50 PCs",
            "Local-neighborhood preservation relative to 50 PCs",
            FIGURE_DIR
            / "02_neighbor_overlap_vs_50pc.png",
        )

        plot_metric_by_dimension(
            classification_table,
            "dimension",
            "balanced_accuracy",
            "Balanced accuracy",
            "Leave-one-individual-out core-state centroid classification",
            FIGURE_DIR
            / "03_core_state_generalization_by_dimension.png",
        )

        plot_metric_by_dimension(
            centroid_geometry,
            "dimension",
            "state_separation_to_individual_dispersion_ratio",
            "State separation / individual dispersion",
            "Biological state separation across PCA dimensions",
            FIGURE_DIR
            / "04_state_separation_ratio_by_dimension.png",
        )

        write_both(
            report,
            "Saved distance, neighborhood, state-generalization, "
            "and state-separation figures.",
        )

        # =========================================================
        # 9. Final status
        # =========================================================

        write_both(
            report,
            "\n9. FINAL STATUS",
        )

        write_both(
            report,
            "-" * 98,
        )

        if (
            passed
            == len(
                checks
            )
            and finite
        ):

            write_both(
                report,
                "FINAL STATUS: PASS",
            )

            write_both(
                report,
                "The transferred 50-PC state space is geometrically "
                "well behaved and biologically reproducible across "
                "independent individuals.",
            )

            write_both(
                report,
                "50 PCs remain frozen as the PRIMARY external-validation "
                "state space.",
            )

            write_both(
                report,
                "5/10/20/30 PCs remain frozen sensitivity analyses.",
            )

            write_both(
                report,
                "The next step may assemble pseudotemporal GDIS windows "
                "without changing dimensionality based on GDIS outcomes.",
            )

        else:

            write_both(
                report,
                "FINAL STATUS: REVIEW REQUIRED",
            )

            write_both(
                report,
                "Review the flagged high-dimensional geometry property "
                "before assembling GDIS trajectories.",
            )

        write_both(
            report,
            "\nIMPORTANT:",
        )

        write_both(
            report,
            "No dimensionality was selected by optimizing a GDIS result.",
        )

        write_both(
            report,
            "No batch correction was introduced.",
        )

        write_both(
            report,
            "NO GDIS VALUES WERE CALCULATED.",
        )

        write_both(
            report,
            f"\nReport:  {REPORT_FILE}",
        )

        write_both(
            report,
            f"Tables:  {TABLE_DIR}",
        )

        write_both(
            report,
            f"Figures: {FIGURE_DIR}",
        )

    print("\n" + "=" * 98)
    print("p16_GSE175634_state_space_geometry_validation.py completed.")
    print("=" * 98)
    print(f"Report : {REPORT_FILE}")
    print(f"Tables : {TABLE_DIR}")
    print(f"Figures: {FIGURE_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

