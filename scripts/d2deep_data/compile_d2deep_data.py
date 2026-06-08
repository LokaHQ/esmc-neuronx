"""Download and compile the D2Deep pathogenicity dataset.

Sources are listed in d2deep_data.csv (Zenodo). Running this script
produces a single cleaned CSV at <data_dir>/compiled_data/compiled_data.csv.

Usage:
    python scripts/d2deep_data/compile_d2deep_data.py
    python scripts/d2deep_data/compile_d2deep_data.py --data_dir /path/to/data
    python scripts/d2deep_data/compile_d2deep_data.py --metadata /path/to/custom_metadata.csv
"""

import argparse
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------
FILENAME_COL = "filename"
DOWNLOAD_URL_COL = "download_url"

UNIPROT_COL = "uniprot"
WT_SEQUENCE_COL = "WT_sequence"
MUT_SEQUENCE_COL = "mut_sequence"
AA_ORIG_COL = "AA_orig"
POSITION_COL = "position"
AA_TARG_COL = "AA_targ"
LABEL_COL = "label"

REQUIRED_COLUMNS = [
    UNIPROT_COL,
    WT_SEQUENCE_COL,
    MUT_SEQUENCE_COL,
    AA_ORIG_COL,
    POSITION_COL,
    AA_TARG_COL,
    LABEL_COL,
]
STRING_COLUMNS = [UNIPROT_COL, WT_SEQUENCE_COL, MUT_SEQUENCE_COL, AA_ORIG_COL, AA_TARG_COL]
NUMERIC_COLUMNS = [POSITION_COL, LABEL_COL]

# ---------------------------------------------------------------------------
# Validation thresholds
# ---------------------------------------------------------------------------
UNIPROT_MIN_LENGTH = 6
UNIPROT_MAX_LENGTH = 10
VALID_LABEL_VALUES = [0, 1]
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
COMPILED_DATA_FILENAME = "compiled_data.csv"
DOWNLOAD_MAX_RETRIES = 10
DOWNLOAD_RETRY_BACKOFF = 2.0  # seconds; doubles on each retry

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def _download_with_retry(url: str, dest: Path) -> bool:
    """Download *url* to *dest*, retrying on 5xx with exponential backoff.

    Returns True on success, False after all retries are exhausted.
    """
    delay = DOWNLOAD_RETRY_BACKOFF
    for attempt in range(1, DOWNLOAD_MAX_RETRIES + 1):
        try:
            # timeout=(connect_s, read_s): 10s to establish, 300s between chunks.
            # stream=True writes incrementally so large files don't stall the socket.
            response = requests.get(url, timeout=(10, 300), stream=True)
            if response.status_code == 200:
                with dest.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=65536):
                        f.write(chunk)
                return True
            if response.status_code < 500:
                print(f"  Failed ({response.status_code}): {url}")
                return False
            # 5xx: server-side transient error — retry
            print(
                f"  Server error ({response.status_code}), retry {attempt}/{DOWNLOAD_MAX_RETRIES} in {delay:.0f}s …"
            )
        except requests.exceptions.Timeout:
            print(f"  Timed out, retry {attempt}/{DOWNLOAD_MAX_RETRIES} in {delay:.0f}s …")
        except Exception as e:
            print(f"  Error: {e}")
            return False
        time.sleep(delay)
        delay *= 2
    print(f"  Gave up after {DOWNLOAD_MAX_RETRIES} attempts: {url}")
    return False


def download_raw_data(metadata_path: Path, raw_data_dir: Path) -> None:
    """Download raw CSV files listed in *metadata_path* into *raw_data_dir*."""
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(metadata_path)
    for _, row in metadata.iterrows():
        save_path = raw_data_dir / row[FILENAME_COL]
        print(f"Downloading: {row[FILENAME_COL]}")
        if _download_with_retry(row[DOWNLOAD_URL_COL], save_path):
            print(f"  Saved: {save_path.name}")


def clean_compiled_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate a mutations DataFrame.

    Steps: drop duplicates, drop nulls/empty strings, enforce dtypes,
    validate UniProt IDs, labels, amino acid codes, sequences, and positions.
    """
    print(f"Starting cleaning — initial shape: {df.shape}")

    df = df.drop_duplicates()
    df = df.dropna()
    df = df.replace(r"^\s*$", pd.NA, regex=True).dropna()
    print(f"After dedup + null removal: {df.shape}")

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in STRING_COLUMNS:
        df[col] = df[col].astype(str).str.strip()
    df = df.dropna()
    print(f"After dtype conversion: {df.shape}")

    df = df[
        df[UNIPROT_COL].apply(
            lambda x: (
                isinstance(x, str)
                and UNIPROT_MIN_LENGTH <= len(x) <= UNIPROT_MAX_LENGTH
                and x.isalnum()
            )
        )
    ]
    print(f"After UniProt validation: {df.shape}")

    df = df[df[LABEL_COL].isin(VALID_LABEL_VALUES)]
    print(f"After label validation: {df.shape}")

    df = df[
        df[AA_ORIG_COL].apply(lambda x: len(str(x)) == 1 and str(x).upper() in AMINO_ACIDS)
        & df[AA_TARG_COL].apply(lambda x: len(str(x)) == 1 and str(x).upper() in AMINO_ACIDS)
    ]
    print(f"After amino acid validation: {df.shape}")

    def valid_sequence(seq):
        return bool(seq) and not pd.isna(seq) and all(aa.upper() in AMINO_ACIDS for aa in str(seq))

    df = df[df[WT_SEQUENCE_COL].apply(valid_sequence) & df[MUT_SEQUENCE_COL].apply(valid_sequence)]
    print(f"After sequence validation: {df.shape}")

    df = df[df[POSITION_COL] > 0]
    df = df[df.apply(lambda r: r[POSITION_COL] <= len(str(r[WT_SEQUENCE_COL])), axis=1)]
    print(f"After position validation: {df.shape}")

    for col in NUMERIC_COLUMNS:
        df[col] = df[col].astype(int)
    for col in [AA_ORIG_COL, AA_TARG_COL, WT_SEQUENCE_COL, MUT_SEQUENCE_COL]:
        df[col] = df[col].str.upper()

    print(f"Cleaning complete — final shape: {df.shape}")
    return df


def compile_all_data(raw_data_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Read all CSVs from *raw_data_dir*, clean, and write a single compiled CSV."""
    csv_files = list(raw_data_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV file(s): {[f.name for f in csv_files]}")

    frames = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
            if missing:
                print(f"Skipping {csv_path.name} — missing columns: {missing}")
                continue
            frames.append(df[REQUIRED_COLUMNS].copy())
            print(f"Loaded {csv_path.name}: {len(df)} rows")
        except Exception as e:
            print(f"Error reading {csv_path.name}: {e}")

    if not frames:
        print("No valid source files found.")
        return pd.DataFrame()

    compiled = pd.concat(frames, ignore_index=True)
    print(f"\nCombined before cleaning: {len(compiled)} rows")
    compiled = clean_compiled_data(compiled)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / COMPILED_DATA_FILENAME
    compiled.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return compiled


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and compile the D2Deep pathogenicity dataset."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="Root directory for raw and compiled data (default: data/)",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=_SCRIPT_DIR / "d2deep_data.csv",
        help="Path to the metadata CSV listing filenames and download URLs",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw_data_dir = args.data_dir / "raw_data"
    compiled_dir = args.data_dir / "compiled_data"

    print(f"Metadata:    {args.metadata}")
    print(f"Raw data:    {raw_data_dir}")
    print(f"Output:      {compiled_dir / COMPILED_DATA_FILENAME}\n")

    download_raw_data(args.metadata, raw_data_dir)
    compile_all_data(raw_data_dir, compiled_dir)


if __name__ == "__main__":
    main()
